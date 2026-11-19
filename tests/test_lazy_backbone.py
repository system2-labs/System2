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
