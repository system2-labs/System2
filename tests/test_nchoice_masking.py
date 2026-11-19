"""test_nchoice_masking.py - Variable N-Choice Encoding, Tensor Masking & Blast-Radius Tests.

Verifies:
1. `mean_pool_pairwise` generalizes cleanly to N = 1..6 choices across standard,
   `span_pooling=True`, and `asym_spans=True`, with strict input validation.
2. `_compute_qa_masks` splits at the LAST ' Answer:' (37741, 27) delimiter and guards T < 2.
3. `System2.forward` preserves bit-for-bit 4-choice math when `choice_mask` is None or all-True.
4. `System2.forward` and `System2Engine.adjudicate` are invariant to padded slot contents
   on both CPU and MPS, handle valid_count == 1 without NaNs/IndexError, and never vote
   for padded indices >= N.
5. Server `/v1/systemone` and `/v1/decisions` handle 5 and 6 choices without IndexError.
"""

import pytest
import torch
from fastapi.testclient import TestClient

from bert import get_bert
from encoding import _compute_qa_masks, mean_pool_pairwise
from sys2.engine import System2Engine
from sys2.server import create_app
from system import System2

DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])


@pytest.fixture(scope="module")
def cpu_engine():
    return System2Engine(device="cpu")


def test_mean_pool_pairwise_variable_n_choices():
    """Verify mean_pool_pairwise supports N = 1..6 across standard, span_pooling, and asym_spans."""
    questions = [
        "Which particle has a negative electric charge?",
        "What is the boiling point of water at 1 atm?",
    ]
    base_choices_q1 = ["Electron", "Proton", "Neutron", "Photon", "Positron", "Gluon"]
    base_choices_q2 = ["100 C", "0 C", "50 C", "212 C", "273 C", "1000 C"]

    for n in (1, 2, 3, 4, 5, 6):
        choices = [base_choices_q1[:n], base_choices_q2[:n]]

        std = mean_pool_pairwise(questions, choices, max_len=64, device="cpu")
        assert std.shape == (2, n, 1024)
        assert torch.isfinite(std).all()

        span = mean_pool_pairwise(
            questions, choices, max_len=64, device="cpu", span_pooling=True
        )
        assert span.shape == (2, n, 1024)
        assert torch.isfinite(span).all()

        asym = mean_pool_pairwise(
            questions,
            choices,
            max_len=64,
            device="cpu",
            asym_spans=True,
            asym_span_mode="choice_forward",
        )
        assert asym.shape == (2, 5, n, 1024)
        assert torch.isfinite(asym).all()


def test_mean_pool_pairwise_validation():
    """Verify clear ValueErrors on empty, mismatched, or ragged inputs."""
    with pytest.raises(ValueError, match="non-empty"):
        mean_pool_pairwise([], [], device="cpu")

    with pytest.raises(ValueError, match="non-empty"):
        mean_pool_pairwise(["q1"], [], device="cpu")

    with pytest.raises(ValueError, match="same length"):
        mean_pool_pairwise(["q1", "q2"], [["a", "b"]], device="cpu")

    with pytest.raises(ValueError, match="at least 1 choice"):
        mean_pool_pairwise(["q1"], [[]], device="cpu")

    with pytest.raises(ValueError, match="same number of choices"):
        mean_pool_pairwise(["q1", "q2"], [["a", "b"], ["c"]], device="cpu")


def test_compute_qa_masks_last_answer_delimiter_and_short_seq():
    """Verify Q/A split picks the FINAL ' Answer:' delimiter and handles T < 2 safely."""
    tok, _ = get_bert()

    prompt_with_distractor = (
        "Question: The passage contains Answer: distractor inside the prompt. "
        "Which option matches? Answer: real choice"
    )
    prompt_clean = (
        "Question: The passage contains prefix without delimiter inside the prompt. "
        "Which option matches? Answer: real choice"
    )
    enc = tok([prompt_with_distractor, prompt_clean], return_tensors="pt", padding=True)
    input_ids = enc["input_ids"]
    attn_mask = enc["attention_mask"]

    q_mask, a_mask = _compute_qa_masks(input_ids, attn_mask)

    # Row 0 has two (37741, 27) occurrences; the Answer span must contain ONLY "real choice",
    # matching row 1's Answer tokens rather than capturing "distractor inside the prompt...".
    row0_ans_ids = input_ids[0][a_mask[0]].tolist()
    row1_ans_ids = input_ids[1][a_mask[1]].tolist()
    assert row0_ans_ids == row1_ans_ids
    assert tok.decode(row0_ans_ids).strip() == "real choice"
    assert "distractor" in tok.decode(input_ids[0][q_mask[0]].tolist())

    # T < 2 guard must not raise RuntimeError on argmax over width-0 slice.
    short_ids = torch.tensor([[50281]], dtype=torch.long)
    short_attn = torch.tensor([[1]], dtype=torch.long)
    sq, sa = _compute_qa_masks(short_ids, short_attn)
    assert torch.equal(sq, short_attn.bool())
    assert torch.equal(sa, short_attn.bool())


@pytest.mark.parametrize("dev", DEVICES)
def test_system2_4choice_regression(dev):
    """Verify choice_mask=None and all-True choice_mask match within atol=1e-6 (bit-for-bit)."""
    configs = [
        dict(
            head_norm_logits=True,
            margin_features=True,
            judger_soft_emb=True,
            decision_conf=True,
            decision_conf_mode="softmax",
        ),
        dict(
            head_center_logits=True,
            consensus_features=True,
            judger_win_emb=True,
            prob_margin_gamma=0.5,
            prob_ensemble=True,
            decision_conf=True,
            decision_conf_mode="margin",
        ),
        dict(
            head_norm_logits=True,
            margin_features=True,
            veto_blend=0.25,
            decision_conf=True,
            decision_conf_mode="hybrid",
        ),
    ]
    for idx, cfg in enumerate(configs):
        torch.manual_seed(10 + idx)
        model = System2(num_heads=5, num_classes=1, pairwise=True, head_layers=2, **cfg).to(dev).eval()
        x = torch.randn(2, 5, 4, 1024, device=dev)
        all_true = torch.ones(2, 4, dtype=torch.bool, device=dev)
        with torch.no_grad():
            out_none = model(x, choice_mask=None)
            out_all_true = model(x, choice_mask=all_true)
        for a, b in zip(out_none, out_all_true):
            assert torch.allclose(a, b, atol=1e-6), f"Regression mismatch on {dev} cfg={idx}"


@pytest.mark.parametrize("dev", DEVICES)
def test_system2_padding_invariance_and_valid_count_one(dev):
    """Verify padded slots never affect valid outputs, cause NaNs, or win votes on CPU and MPS."""
    configs = [
        dict(
            head_norm_logits=True,
            margin_features=True,
            judger_soft_emb=True,
            head_weights=[1.0, 1.0, 1.1, 1.0, 0.5],
            decision_conf=True,
            decision_conf_mode="softmax",
        ),
        dict(
            head_center_logits=True,
            consensus_features=True,
            judger_win_emb=True,
            prob_margin_gamma=0.4,
            prob_ensemble=True,
            decision_conf=True,
            decision_conf_mode="margin",
        ),
        dict(
            head_norm_logits=True,
            margin_features=True,
            veto_blend=0.2,
            dual_stream_contest=True,
            decision_conf=True,
            decision_conf_mode="hybrid",
        ),
        dict(uniform_ensemble=True, diff_features=False, margin_features=False),
    ]

    for idx, cfg in enumerate(configs):
        torch.manual_seed(100 + idx)
        model = System2(num_heads=5, num_classes=1, pairwise=True, head_layers=2, **cfg).to(dev).eval()
        x = torch.randn(3, 5, 4, 1024, device=dev)
        # Mixed batch: valid counts N=1, N=2, N=3
        choice_mask = torch.tensor(
            [
                [True, False, False, False],
                [True, True, False, False],
                [True, True, True, False],
            ],
            device=dev,
        )
        x_garbage = x.clone()
        x_garbage[~choice_mask.unsqueeze(1).expand(-1, 5, -1)] = 987.65

        with torch.no_grad():
            out_clean = model(x, choice_mask=choice_mask)
            out_garb = model(x_garbage, choice_mask=choice_mask)

        for t_idx, (t_clean, t_garb) in enumerate(zip(out_clean, out_garb)):
            assert torch.isfinite(t_clean).all(), f"Non-finite out[{t_idx}] on {dev} cfg={idx}"
            assert torch.isfinite(t_garb).all(), f"Non-finite garbage out[{t_idx}] on {dev} cfg={idx}"
            assert torch.allclose(t_clean, t_garb, atol=1e-5), (
                f"Padding leakage in out[{t_idx}] on {dev} cfg={idx}: "
                f"max_diff={(t_clean - t_garb).abs().max().item()}"
            )

        logits = out_clean[4]      # (3, 5, 4)
        ens_logits = out_clean[7]  # (3, 4)
        assert (logits[~choice_mask.unsqueeze(1).expand(-1, 5, -1)] == -1e9).all()
        assert (ens_logits[~choice_mask] == -1e9).all()

        head_votes = logits.argmax(dim=-1)
        ens_wins = ens_logits.argmax(dim=-1)
        for row_i, n_valid in enumerate((1, 2, 3)):
            assert (head_votes[row_i] < n_valid).all()
            assert ens_wins[row_i].item() < n_valid

        # Unpadded 1-choice tensor (B, 5, 1, 1024) must also run with zero NaN and no IndexError.
        x_one = torch.randn(2, 5, 1, 1024, device=dev)
        with torch.no_grad():
            out_one = model(x_one)
        for t_one in out_one:
            assert torch.isfinite(t_one).all()

    # Invalid choice_mask shapes or all-False rows must fail loudly.
    model_cpu = System2(num_heads=5, num_classes=1, pairwise=True, head_layers=2).eval()
    with pytest.raises(ValueError, match="does not match"):
        model_cpu(torch.randn(1, 5, 4, 1024), choice_mask=torch.ones(1, 3, dtype=torch.bool))
    with pytest.raises(ValueError, match="at least one True"):
        model_cpu(torch.randn(1, 5, 4, 1024), choice_mask=torch.zeros(1, 4, dtype=torch.bool))


def test_engine_adjudicate_empty_raises(cpu_engine):
    """Empty choices list must raise ValueError immediately."""
    with pytest.raises(ValueError, match="non-empty"):
        cpu_engine.adjudicate("What is 2 + 2?", [])


@pytest.mark.parametrize("dev", DEVICES)
def test_engine_padding_invariance_and_nchoice_end_to_end(dev):
    """Prove end-to-end padding invariance (N=1..3) and N=5..6 execution on CPU and MPS."""
    tok, shared_mod = get_bert()
    orig_dev = next(shared_mod.parameters()).device
    try:
        eng = System2Engine(device=dev)
        eng.switch_domain("science")
        question = "What state of matter is water at 20 degrees Celsius?"
        all_candidates = [
            "Liquid",
            "Solid",
            "Gas",
            "Plasma",
            "Bose-Einstein condensate",
            "Superfluid",
        ]

        # 1. N = 1, 2, 3: replace "N/A" padding with arbitrary garbage text and verify
        # valid logits, probabilities, head votes, and confidence match within atol=1e-5.
        for n in (1, 2, 3):
            valid = all_candidates[:n]
            res = eng.adjudicate(question, valid, domain="science")
            assert 0 <= res.selected_index < n
            assert all(0 <= h < n for h in res.per_head_choices)
            assert res.choice_probabilities is not None
            assert len(res.choice_probabilities) == n
            assert sum(res.choice_probabilities) == pytest.approx(1.0, abs=0.01)

            pad_na = valid + ["N/A"] * (4 - n)
            pad_garbage = valid + [f"UNRELATED_GARBAGE_FILLER_{i} " * 6 for i in range(4 - n)]
            mask = torch.zeros((1, 4), dtype=torch.bool, device=dev)
            mask[:, :n] = True

            with torch.no_grad():
                vecs_na = mean_pool_pairwise(
                    [question],
                    [pad_na],
                    max_len=256,
                    device=dev,
                    asym_spans=True,
                    asym_span_mode="choice_forward",
                    model=eng.model,
                    tokenizer=eng.tokenizer,
                )
                vecs_garb = mean_pool_pairwise(
                    [question],
                    [pad_garbage],
                    max_len=256,
                    device=dev,
                    asym_spans=True,
                    asym_span_mode="choice_forward",
                    model=eng.model,
                    tokenizer=eng.tokenizer,
                )
                out_na = eng.current_sys2(vecs_na, choice_mask=mask)
                out_garb = eng.current_sys2(vecs_garb, choice_mask=mask)

            assert torch.allclose(out_na[4][:, :, :n], out_garb[4][:, :, :n], atol=1e-5)
            assert torch.allclose(out_na[7][:, :n], out_garb[7][:, :n], atol=1e-5)
            assert torch.allclose(out_na[1], out_garb[1], atol=1e-5)

            probs_na = torch.softmax(out_na[7][:, :n], dim=-1)
            probs_garb = torch.softmax(out_garb[7][:, :n], dim=-1)
            assert torch.allclose(probs_na, probs_garb, atol=1e-5)

            heads_na = out_na[4].squeeze(0).argmax(dim=-1).tolist()
            heads_garb = out_garb[4].squeeze(0).argmax(dim=-1).tolist()
            assert heads_na == heads_garb
            assert all(0 <= h < n for h in heads_na)

        # 2. N = 5 and N = 6: no truncation, full probability vector returned.
        for n in (5, 6):
            res = eng.adjudicate(question, all_candidates[:n], domain="science")
            assert 0 <= res.selected_index < n
            assert res.selected_choice == all_candidates[res.selected_index]
            assert all(0 <= h < n for h in res.per_head_choices)
            assert res.choice_probabilities is not None
            assert len(res.choice_probabilities) == n
            assert sum(res.choice_probabilities) == pytest.approx(1.0, abs=0.01)
    finally:
        # Restore the shared cached ModernBERT backbone device so subsequent CPU test
        # fixtures do not incur cross-device moves mid-suite.
        shared_mod.to(orig_dev)


def test_server_5_and_6_choices_systemone_and_decisions(cpu_engine):
    """Verify /v1/systemone and /v1/decisions handle 5 and 6 choices with HTTP 200 and no IndexError."""
    app = create_app(engine=cpu_engine, device="cpu")
    client = TestClient(app)

    # 1. /v1/systemone: 5-choice 'choice' + 6-level 'score'
    sysone_payload = {
        "model": "system2",
        "state": "Water at 20 degrees Celsius and 1 atmosphere of pressure.",
        "questions": {
            "phase_choice_5": {
                "type": "choice",
                "instructions": "Select the phase of water.",
                "criteria": {
                    "liquid": "Fluid liquid water",
                    "solid": "Frozen ice crystal",
                    "gas": "Water vapor steam",
                    "plasma": "Ionized plasma state",
                    "supercritical": "Supercritical fluid",
                },
            },
            "phase_score_6": {
                "type": "score",
                "instructions": "Rate the thermal energy state from lowest to highest.",
                "criteria": [
                    "Absolute zero",
                    "Frozen solid",
                    "Cold liquid",
                    "Warm liquid",
                    "Boiling vapor",
                    "Ionized plasma",
                ],
            },
        },
    }
    resp1 = client.post("/v1/systemone", json=sysone_payload)
    assert resp1.status_code == 200, resp1.text
    d1 = resp1.json()
    c_ans = d1["answers"]["phase_choice_5"]
    assert c_ans["choice"] in sysone_payload["questions"]["phase_choice_5"]["criteria"]
    assert len(c_ans["probabilities"]) == 5
    assert sum(c_ans["probabilities"].values()) == pytest.approx(1.0, abs=0.01)

    s_ans = d1["answers"]["phase_score_6"]
    assert len(s_ans["probabilities"]) == 6
    assert sum(s_ans["probabilities"]) == pytest.approx(1.0, abs=0.01)
    assert 0.0 <= s_ans["score"] <= 5.0

    # 2. /v1/decisions: 5-choice 'choice' + 6-level 'score'
    dec_payload = {
        "model": "system2:science",
        "input": "Water at 20 degrees Celsius and 1 atmosphere of pressure.",
        "questions": [
            {
                "type": "choice",
                "name": "phase_choice_5",
                "instructions": "Select the phase of water.",
                "choices": [
                    {"value": "liquid", "description": "Fluid liquid water"},
                    {"value": "solid", "description": "Frozen ice crystal"},
                    {"value": "gas", "description": "Water vapor steam"},
                    {"value": "plasma", "description": "Ionized plasma state"},
                    {"value": "bose", "description": "Bose-Einstein condensate"},
                ],
            },
            {
                "type": "score",
                "name": "phase_score_6",
                "instructions": "Rate thermal energy state.",
                "levels": [
                    {"label": "Level 0: Absolute zero"},
                    {"label": "Level 1: Frozen solid"},
                    {"label": "Level 2: Cold liquid"},
                    {"label": "Level 3: Warm liquid"},
                    {"label": "Level 4: Boiling vapor"},
                    {"label": "Level 5: Ionized plasma"},
                ],
            },
        ],
    }
    resp2 = client.post("/v1/decisions", json=dec_payload)
    assert resp2.status_code == 200, resp2.text
    d2 = resp2.json()
    assert len(d2["answers"]) == 2
    assert d2["answers"][0]["choice"] in {"liquid", "solid", "gas", "plasma", "bose"}
    assert 0.0 <= d2["answers"][1]["score"] <= 5.0
