"""test_lazy_backbone.py - ModernBERT must load lazily and exactly once.

Guards the t_9a8a5d25 fix: importing core modules never loads the 395M backbone,
System2Engine + DomainRouter + bert.get_bert() share ONE cached model object, and
the router's centroid embeddings stay LoRA-free even though they now run on the
LoRA-adapted shared backbone.

Heavy checks run in fresh subprocesses so they cannot be polluted by (or pollute)
modules other tests already imported into the pytest process.
"""

import json
import os
import subprocess
import sys
import textwrap
import time

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _run_py(code: str, timeout: int = 600) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_imports_are_lazy_and_fast():
    # (a) Importing every module that used to trigger an eager from_pretrained must
    # leave the cache empty and stay fast. torch is imported BEFORE the timer: it is
    # an unavoidable ~1-2s third-party cost, whereas the repo's own imports used to
    # add a ~10-16s ModernBERT load on top.
    code = """
        import json, time
        import torch  # noqa: F401
        t0 = time.perf_counter()
        import bert, heads, system, encoding, load_system2
        import sys2.engine, routing.domain_router
        print(json.dumps({"cached": sorted(bert._cached_models), "secs": time.perf_counter() - t0}))
    """
    proc = _run_py(code)
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out["cached"] == [], f"backbone loaded eagerly: {out['cached']}"
    assert out["secs"] < 2.0, f"imports took {out['secs']:.2f}s"


def test_get_bert_import_does_not_load():
    # `from bert import get_bert` must not touch the cache either (legacy-compat
    # __getattr__ only fires on tokenizer/model attribute access).
    proc = _run_py(
        """
        from bert import get_bert
        import bert
        print(bert._cached_models == {})
        """
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().splitlines()[-1] == "True"


@pytest.fixture(scope="module")
def shared_backbone_probe():
    # (b) + (c) share one subprocess: building the engine loads the 395M model, which
    # is the dominant cost, so do it once.
    code = """
        import json
        import torch
        import bert
        from transformers import AutoModel

        calls = {"n": 0}
        orig = AutoModel.from_pretrained
        def counting(*a, **k):
            calls["n"] += 1
            return orig(*a, **k)
        AutoModel.from_pretrained = counting

        from sys2.engine import System2Engine
        from bert import get_bert
        from lora import LoRALinear

        engine = System2Engine(device="cpu")
        router = engine.router
        text = "Why do plants appear green under sunlight?"

        shared = (
            engine.model is router.bert_model is get_bert()[1]
            and engine.tokenizer is router.tokenizer is get_bert()[0]
        )

        before = router.encode_text(text)

        # Give every LoRA B a non-zero value so the delta is clearly visible.
        g = torch.Generator().manual_seed(0)
        n_lora = 0
        with torch.no_grad():
            for m in engine.model.modules():
                if isinstance(m, LoRALinear):
                    m.lora_B.copy_(torch.randn(m.lora_B.shape, generator=g) * 0.5)
                    n_lora += 1
        after = router.encode_text(text)

        # Sanity: with LoRA enabled the raw backbone output really does differ, so the
        # equality above is meaningful rather than vacuous.
        from encoding import mean_pool_texts
        with_lora = mean_pool_texts([text], device="cpu", model=engine.model, tokenizer=engine.tokenizer)
        from lora import disable_lora
        with disable_lora(engine.model):
            without = mean_pool_texts([text], device="cpu", model=engine.model, tokenizer=engine.tokenizer)
        states_restored = all(m.enabled for m in engine.model.modules() if isinstance(m, LoRALinear))

        print(json.dumps({
            "from_pretrained_calls": calls["n"],
            "shared": bool(shared),
            "n_lora": n_lora,
            "router_stable": bool(torch.allclose(before, after, atol=1e-5)),
            "lora_changes_output": bool(not torch.allclose(with_lora, without, atol=1e-3)),
            "enabled_restored": bool(states_restored),
            "out_device": str(after.device),
        }))
    """
    proc = _run_py(code)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_engine_router_share_single_backbone(shared_backbone_probe):
    assert shared_backbone_probe["shared"]
    assert shared_backbone_probe["from_pretrained_calls"] == 1


def test_disable_lora_keeps_router_embeddings_pure(shared_backbone_probe):
    p = shared_backbone_probe
    assert p["n_lora"] > 0
    assert p["lora_changes_output"], "test is vacuous: LoRA did not alter backbone output"
    assert p["router_stable"], "router embedding drifted after LoRA adapter load"
    assert p["enabled_restored"]
    assert p["out_device"] == "cpu"


def test_legacy_module_attributes_still_work():
    # (d) Legacy scripts (eval.py/train.py/interact.py) do `from bert import tokenizer, model`.
    # Use a tiny stand-in loader so this stays fast: patch the HF constructors.
    code = """
        import bert
        import transformers
        class _Tok:  # minimal stand-ins; we only assert wiring, not model quality
            pass
        class _Mod:
            pass
        transformers.AutoTokenizer.from_pretrained = staticmethod(lambda *a, **k: _Tok())
        transformers.AutoModel.from_pretrained = staticmethod(lambda *a, **k: _Mod())
        assert bert._cached_models == {}
        from bert import tokenizer, model
        t, m = bert.get_bert()
        assert tokenizer is t and model is m
        assert len(bert._cached_models) == 1
        try:
            bert.does_not_exist
        except AttributeError:
            print("ok")
        """
    proc = _run_py(code)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().splitlines()[-1] == "ok"


def test_import_timing_sanity():
    # Cheap guard that `import bert` alone stays far below the old ~16s eager load.
    t0 = time.perf_counter()
    proc = _run_py("import bert")
    assert proc.returncode == 0, proc.stderr
    assert time.perf_counter() - t0 < 10.0


def test_disable_lora_thread_isolation_and_nesting():
    # Pure unit test covering thread-local isolation, overlap, nesting, exception
    # cleanup, explicit `.enabled = False` preservation, and deepcopy safety.
    import copy
    import threading
    import torch
    import torch.nn as nn
    from lora import LoRALinear, disable_lora

    torch.manual_seed(0)
    base_lin = nn.Linear(16, 16)
    lora_lin = LoRALinear(base_lin, r=4, alpha=8)
    with torch.no_grad():
        lora_lin.lora_A.normal_(0, 0.5)
        lora_lin.lora_B.normal_(0, 0.5)
    model = nn.Sequential(lora_lin)
    x = torch.randn(3, 16)

    with torch.no_grad():
        base_out = base_lin(x)
        lora_out = model(x)
    assert not torch.allclose(base_out, lora_out, atol=1e-3), "test is vacuous"

    # Nesting + exception safety
    with disable_lora(model):
        with disable_lora(model):
            assert torch.allclose(model(x), base_out, atol=1e-6)
        # Inner exited, outer still active -> must remain bypassed
        assert torch.allclose(model(x), base_out, atol=1e-6)
    assert torch.allclose(model(x), lora_out, atol=1e-6)

    with pytest.raises(RuntimeError):
        with disable_lora(model):
            assert torch.allclose(model(x), base_out, atol=1e-6)
            raise RuntimeError("boom")
    assert torch.allclose(model(x), lora_out, atol=1e-6)

    # Explicit enabled=False survives disable_lora entry/exit
    lora_lin.enabled = False
    with disable_lora(model):
        assert torch.allclose(model(x), base_out, atol=1e-6)
    assert lora_lin.enabled is False
    assert torch.allclose(model(x), base_out, atol=1e-6)
    lora_lin.enabled = True

    # Overlapping disable_lora across two threads: T1 exits while T2 is inside;
    # T2 must stay disabled until T2 exits, and shared `.enabled` must stay True.
    entered = threading.Barrier(2)
    t1_exited = threading.Event()
    results = {}

    def t1():
        with disable_lora(model):
            entered.wait(timeout=5)
            results["t1_inside"] = model(x).detach()
        t1_exited.set()

    def t2():
        with disable_lora(model):
            entered.wait(timeout=5)
            assert t1_exited.wait(timeout=5)
            results["t2_after_t1_exit"] = model(x).detach()

    th1 = threading.Thread(target=t1)
    th2 = threading.Thread(target=t2)
    th1.start()
    th2.start()
    th1.join(timeout=5)
    th2.join(timeout=5)
    assert torch.allclose(results["t1_inside"], base_out, atol=1e-6)
    assert torch.allclose(results["t2_after_t1_exit"], base_out, atol=1e-6)
    assert lora_lin.enabled is True
    assert torch.allclose(model(x), lora_out, atol=1e-6)

    # Cross-thread isolation: T1 inside disable_lora must not disable LoRA for T2.
    both_ready = threading.Barrier(2)
    t2_done = threading.Event()

    def disabled_thread():
        with disable_lora(model):
            both_ready.wait(timeout=5)
            results["disabled"] = model(x).detach()
            assert t2_done.wait(timeout=5)

    def enabled_thread():
        both_ready.wait(timeout=5)
        results["enabled"] = model(x).detach()
        t2_done.set()

    th1 = threading.Thread(target=disabled_thread)
    th2 = threading.Thread(target=enabled_thread)
    th1.start()
    th2.start()
    th1.join(timeout=5)
    th2.join(timeout=5)
    assert torch.allclose(results["disabled"], base_out, atol=1e-6)
    assert torch.allclose(results["enabled"], lora_out, atol=1e-6)

    # Module remains deepcopy-able (TLS lives at module level, not on instances).
    cloned = copy.deepcopy(lora_lin)
    assert torch.allclose(cloned(x), lora_out, atol=1e-6)


def test_get_bert_cold_cache_is_thread_safe():
    # Concurrent first calls to get_bert() must invoke from_pretrained only once.
    code = """
        import threading, time, json
        import bert, transformers

        calls = {"tok": 0, "mod": 0}
        class _Tok:
            pass
        class _Mod:
            pass

        def fake_tok(*a, **k):
            time.sleep(0.05)
            calls["tok"] += 1
            return _Tok()

        def fake_mod(*a, **k):
            time.sleep(0.05)
            calls["mod"] += 1
            return _Mod()

        transformers.AutoTokenizer.from_pretrained = staticmethod(fake_tok)
        transformers.AutoModel.from_pretrained = staticmethod(fake_mod)

        barrier = threading.Barrier(4)
        out = [None] * 4
        def worker(i):
            barrier.wait(timeout=5)
            out[i] = bert.get_bert()

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        same_tok = all(o[0] is out[0][0] for o in out)
        same_mod = all(o[1] is out[0][1] for o in out)
        print(json.dumps({"calls": calls, "same_tok": same_tok, "same_mod": same_mod}))
    """
    proc = _run_py(code)
    assert proc.returncode == 0, proc.stderr
    res = json.loads(proc.stdout.strip().splitlines()[-1])
    assert res["calls"] == {"tok": 1, "mod": 1}
    assert res["same_tok"] and res["same_mod"]


def test_concurrent_server_route_and_deliberate():
    # Guards the live FastAPI server against both the concurrent /route LoRA
    # stuck-disabled bug and unserialized /route vs /deliberate races (including
    # cold-start initialization).
    code = """
        import asyncio, json
        import httpx, torch
        from transformers import AutoModel
        from sys2.engine import System2Engine
        from sys2.server import create_app
        from lora import LoRALinear

        calls = {"n": 0}
        orig = AutoModel.from_pretrained
        def counting(*a, **k):
            calls["n"] += 1
            return orig(*a, **k)
        AutoModel.from_pretrained = counting

        prompt = "Explain why leaves change color in autumn during shorter days."
        delib_payload = {
            "question": "Why do plants appear green under sunlight?",
            "choices": [
                "Chlorophyll reflects green light wavelengths.",
                "Mitochondria emit green photons.",
            ],
            "domain": "science",
        }

        async def run():
            # 1. Cold server: concurrent /route and /deliberate initialize lazily under lock.
            cold_app = create_app(device="cpu")
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=cold_app), base_url="http://t"
            ) as c:
                resps = await asyncio.gather(
                    *[c.post("/route", json={"prompt": prompt}) for _ in range(2)],
                    *[c.post("/deliberate", json=delib_payload) for _ in range(2)],
                )
                assert all(r.status_code == 200 for r in resps), [r.status_code for r in resps]

            # 2. Pre-built engine: concurrent /route + /deliberate reuse engine.router and
            #    leave zero LoRALinear layers stuck disabled.
            eng = System2Engine(device="cpu")
            from routing.domain_router import DomainRouter
            router_inits = {"n": 0}
            orig_rtr_init = DomainRouter.__init__
            def counting_rtr_init(self, *a, **k):
                router_inits["n"] += 1
                return orig_rtr_init(self, *a, **k)
            DomainRouter.__init__ = counting_rtr_init

            warm_app = create_app(engine=eng, device="cpu")
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=warm_app), base_url="http://t"
            ) as c:
                resps = await asyncio.gather(
                    *[c.post("/route", json={"prompt": prompt}) for _ in range(4)],
                    *[c.post("/deliberate", json=delib_payload) for _ in range(2)],
                )
                assert all(r.status_code == 200 for r in resps), [r.status_code for r in resps]

            stuck = sum(not m.enabled for m in eng.model.modules() if isinstance(m, LoRALinear))
            return {
                "from_pretrained_calls": calls["n"],
                "stuck": stuck,
                "warm_router_inits": router_inits["n"],
            }

        print(json.dumps(asyncio.run(run())))
    """
    proc = _run_py(code)
    assert proc.returncode == 0, proc.stderr
    res = json.loads(proc.stdout.strip().splitlines()[-1])
    assert res["from_pretrained_calls"] == 1
    assert res["stuck"] == 0
    assert res["warm_router_inits"] == 0


def test_bert_encode_follows_backbone_device():
    # Guards against CPU/MPS device mismatch when bert.encode() or encoding helpers
    # run after System2Engine places the shared backbone on MPS (falls back to CPU
    # check on machines without MPS).
    code = """
        import json, torch
        import bert
        from encoding import mean_pool_texts
        from sys2.engine import System2Engine

        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        System2Engine(device=dev)
        v1 = bert.encode("hello world")
        v2 = mean_pool_texts(["hello world"], device="cpu")
        print(json.dumps({
            "dev": dev,
            "v1_shape": list(v1.shape),
            "v1_dev_type": v1.device.type,
            "v2_shape": list(v2.shape),
            "v2_dev_type": v2.device.type,
        }))
    """
    proc = _run_py(code)
    assert proc.returncode == 0, proc.stderr
    res = json.loads(proc.stdout.strip().splitlines()[-1])
    assert res["v1_shape"] == [1, 1024]
    assert res["v1_dev_type"] == res["dev"]
    assert res["v2_shape"] == [1, 1024]
    assert res["v2_dev_type"] == "cpu"
