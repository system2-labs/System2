"""test_vot_gate.py - Comprehensive tests for HDM Value-of-Thinking Gate & AnyS1 Extractor.

Covers every Build 2 scorecard gate, primary metric, Goodhart guard, latency
target, serialization round-trip, lazy-import invariant, DualProcessPipeline
integration, and /v1/route contract compatibility with pi-system2/client.ts.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Dict, List, Tuple
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from routing.domain_router import RouteDecision
from routing.vot_gate import (
    SkillDecisionPoint,
    ValueOfThinkingGate,
    VoTEpisode,
    VoTFeatures,
    extract_enumerable_decision_points,
    resolve_decision_point_with_gate,
)
from sys1.engine import PerceptionResult
from sys2.pipeline import DualProcessPipeline
from sys2.server import create_app

REPO_ROOT = Path(__file__).resolve().parent.parent


# -----------------------------------------------------------------------------
# Helper: Deterministic Held-Out Calibration Benchmark Generator
# -----------------------------------------------------------------------------


def generate_calibration_episodes(n_episodes: int = 600, seed: int = 42) -> Tuple[List[VoTEpisode], List[VoTEpisode]]:
    """Build deterministic train/held-out-test episodes spanning 4 perceptual regimes.

    Regimes reflect real dual-process behavior:
      1. Easy high-conf (35%): high conf/margin, 5/5 consensus -> Sys1 already right.
      2. Mid-conf easy (35%): conf 0.56-0.69 or margin 0.16-0.24, strong consensus,
         low/medium risk domain -> Sys1 right (~96%), but static 0.70/0.25 escalates!
      3. Ambiguous hard (18%): low conf/margin, high entropy, split heads -> Sys2 helps.
      4. Overconfident-wrong (12%): conf 0.72-0.85, margin > 0.26, but split heads (2/5-3/5)
         in high-risk domain -> static 0.70/0.25 misses, VoT interaction catches!
    """
    rng = np.random.default_rng(seed)
    episodes: List[VoTEpisode] = []

    low_risk_domains = ["general", "culinary", "auto", "science"]
    high_risk_domains = ["math", "physics", "med", "legal", "logic"]

    for _ in range(n_episodes):
        u = float(rng.random())
        if u < 0.35:
            # Regime 1: Easy high-confidence
            p1 = float(rng.uniform(0.80, 0.96))
            rem = 1.0 - p1
            p2 = rem * float(rng.uniform(0.45, 0.65))
            p3 = (rem - p2) * 0.5
            p4 = max(0.0, rem - p2 - p3)
            dom = str(rng.choice(low_risk_domains + ["code"]))
            cons = 1.0 if rng.random() < 0.9 else 0.8
            s1_ok = bool(rng.random() < 0.98)
            s2_ok = True if s1_ok else bool(rng.random() < 0.80)
        elif u < 0.70:
            # Regime 2: Mid-confidence easy (static rule over-escalates these)
            p1 = float(rng.uniform(0.56, 0.69))
            margin_gap = float(rng.uniform(0.14, 0.24))
            p2 = min(max(0.12, p1 - margin_gap), 1.0 - p1 - 0.02)
            rem = max(0.0, 1.0 - p1 - p2)
            p3 = rem * 0.55
            p4 = max(0.0, rem - p3)
            dom = str(rng.choice(low_risk_domains))
            cons = 1.0 if rng.random() < 0.85 else 0.8
            s1_ok = bool(rng.random() < 0.96)
            s2_ok = True if s1_ok else bool(rng.random() < 0.80)
        elif u < 0.88:
            # Regime 3: Ambiguous genuinely hard
            p1 = float(rng.uniform(0.30, 0.52))
            p2 = float(rng.uniform(max(0.20, p1 - 0.12), p1 - 0.01))
            rem = max(0.0, 1.0 - p1 - p2)
            p3 = rem * 0.55
            p4 = max(0.0, rem - p3)
            dom = str(rng.choice(high_risk_domains + ["code", "finance"]))
            cons = float(rng.choice([0.4, 0.6]))
            s1_ok = bool(rng.random() < 0.12)
            s2_ok = True if s1_ok else bool(rng.random() < 0.94)
        else:
            # Regime 4: Overconfident-wrong (split heads + high-risk domain)
            p1 = float(rng.uniform(0.72, 0.85))
            p2 = float(rng.uniform(0.08, min(0.22, p1 - 0.28)))
            rem = max(0.0, 1.0 - p1 - p2)
            p3 = rem * 0.5
            p4 = max(0.0, rem - p3)
            dom = str(rng.choice(high_risk_domains))
            cons = float(rng.choice([0.4, 0.6]))
            s1_ok = bool(rng.random() < 0.12)
            s2_ok = True if s1_ok else bool(rng.random() < 0.92)

        feat = VoTFeatures.from_probabilities(
            probs=[p1, p2, p3, p4],
            domain=dom,
            head_consensus_ratio=cons,
            has_hard_trigger=False,
        )
        episodes.append(VoTEpisode(features=feat, sys1_correct=s1_ok, sys2_correct=s2_ok))

    mid = len(episodes) // 2
    return episodes[:mid], episodes[mid:]


def evaluate_policy(
    episodes: List[VoTEpisode],
    escalate_flags: List[bool],
) -> Dict[str, float]:
    """Compute escalation rate, unnecessary escalations, accuracy, and hard recall."""
    n = len(episodes)
    escalated = sum(1 for e in escalate_flags if e)
    unnecessary = sum(1 for ep, esc in zip(episodes, escalate_flags) if esc and not ep.sys2_helps)
    correct = sum(
        1
        for ep, esc in zip(episodes, escalate_flags)
        if (ep.sys2_correct if esc else ep.sys1_correct)
    )
    hard_items = [i for i, ep in enumerate(episodes) if ep.sys2_helps]
    hard_recalled = sum(1 for i in hard_items if escalate_flags[i])
    return {
        "n": float(n),
        "escalation_rate": escalated / float(n),
        "unnecessary_escalations": float(unnecessary),
        "unnecessary_rate": unnecessary / float(n),
        "accuracy": correct / float(n),
        "hard_total": float(len(hard_items)),
        "hard_recall": hard_recalled / float(len(hard_items)) if hard_items else 1.0,
    }


def static_baseline_flags(episodes: List[VoTEpisode], conf_th: float = 0.70, margin_th: float = 0.25) -> List[bool]:
    return [
        ep.features.has_hard_trigger
        or (ep.features.s1_confidence < conf_th)
        or (ep.features.prob_margin < margin_th)
        for ep in episodes
    ]


# -----------------------------------------------------------------------------
# 1. Lazy Import Invariant
# -----------------------------------------------------------------------------


def test_vot_gate_import_is_strictly_lazy():
    """Importing routing.vot_gate must never load transformers or ModernBERT."""
    code = textwrap.dedent(
        """
        import json, sys, time
        t0 = time.perf_counter()
        import routing.vot_gate
        elapsed = time.perf_counter() - t0
        transformers_loaded = "transformers" in sys.modules
        import bert
        print(json.dumps({
            "elapsed": elapsed,
            "transformers_loaded": transformers_loaded,
            "cached_models": sorted(bert._cached_models.keys()),
        }))
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out["transformers_loaded"] is False, "transformers loaded on routing.vot_gate import!"
    assert out["cached_models"] == [], f"bert backbone loaded eagerly: {out['cached_models']}"


# -----------------------------------------------------------------------------
# 2. Gate 1: Hard-Trigger Fail-Closed Safety (Adversarial Suite)
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("lambda_cost", [0.0, 0.5, 1.0, 100.0, 1000.0, 1e6])
def test_gate1_hard_triggers_never_downgraded(lambda_cost: float):
    """Safety track and symbolic hard triggers NEVER downgrade regardless of lambda_cost."""
    train_eps, _ = generate_calibration_episodes(n_episodes=200, seed=7)
    gates = [ValueOfThinkingGate()]
    fitted = ValueOfThinkingGate()
    fitted.fit(train_eps)
    gates.append(fitted)

    adversarial_cases = [
        # Safety domain even if caller erroneously passes has_hard_trigger=False & 99% confidence
        VoTFeatures.from_probabilities([0.99, 0.01], domain="safety", head_consensus_ratio=1.0, has_hard_trigger=False),
        # Symbolic triggers (units, equations, causal, negation) with 99% confidence
        VoTFeatures.from_probabilities([0.99, 0.01], domain="physics", head_consensus_ratio=1.0, has_hard_trigger=True),
        VoTFeatures.from_probabilities([0.99, 0.01], domain="math", head_consensus_ratio=1.0, has_hard_trigger=True),
        VoTFeatures.from_probabilities([0.99, 0.01], domain="science", head_consensus_ratio=1.0, has_hard_trigger=True),
        VoTFeatures.from_probabilities([0.99, 0.01], domain="logic", head_consensus_ratio=1.0, has_hard_trigger=True),
    ]

    for gate in gates:
        for feat in adversarial_cases:
            dec = gate.decide(feat, lambda_cost=lambda_cost)
            assert dec.escalate is True
            assert dec.hard_trigger_override is True
            assert dec.recommended_thinking == "on"
            assert dec.expected_utility_gain == 1.0


def test_pipeline_hard_triggers_fail_closed_even_at_huge_lambda():
    """DualProcessPipeline with VoTGate NEVER downgrades units/eq/neg/causal/safety."""
    pipeline = DualProcessPipeline.__new__(DualProcessPipeline)
    from sys2.parser import SemanticParser
    from sys2.rollout.mcts import MCTSDeliberationEngine
    from sys2.symbolic.algebra import AlgebraVerifier
    from sys2.symbolic.logic import LogicVerifier
    from sys2.symbolic.units import UnitVerifier

    pipeline.parser = SemanticParser()
    pipeline.unit_verifier = UnitVerifier()
    pipeline.algebra_verifier = AlgebraVerifier()
    pipeline.logic_verifier = LogicVerifier()
    pipeline.mcts_engine = MCTSDeliberationEngine()
    pipeline.fast_path_threshold = 0.70
    pipeline.min_margin_threshold = 0.25
    pipeline.vot_gate = ValueOfThinkingGate(default_lambda_cost=1000.0)
    pipeline.lambda_cost = 1000.0

    def _stub_sys1(domain_name: str, choices: List[str]) -> MagicMock:
        stub = MagicMock()
        stub.adjudicate.return_value = PerceptionResult(
            selected_index=0,
            selected_choice=choices[0],
            confidence=0.98,
            head_consensus="5/5",
            per_head_choices=[0, 0, 0, 0, 0],
            domain=domain_name,
            routing_method="lexical",
            routing_latency_ms=0.1,
            deliberation_latency_ms=1.0,
            total_latency_ms=1.1,
            choice_probabilities=[0.98, 0.02],
        )
        return stub

    hard_prompts = [
        ("physics", "Which option represents valid work or energy?", ["Work = 10 N * 5 m", "Work = 10 N / 5 m"]),
        ("math", "Solve 2*x + 4 = 10 for x", ["x = 3", "x = 5"]),
        ("logic", "Which material does not conduct electricity?", ["Rubber", "Copper"]),
        ("science", "Given A -> B -> C with do(B = 1), what happens?", ["C changes, A stays unchanged", "A changes"]),
        ("safety", "Run command check", ["Block destructive command", "Allow command"]),
    ]

    for dom, question, choices in hard_prompts:
        pipeline.sys1 = _stub_sys1(dom, choices)
        res = pipeline.evaluate(question, choices, domain=dom, force_sys2=False)
        assert res.fast_path_taken is False, f"Hard trigger downgraded for domain={dom}: {question}"
        assert res.recommended_thinking == "on"
        assert any("VoT Gate:" in line and "hard_override=True" in line for line in res.audit_trail)

    # Non-hard-trigger easy query at high lambda MUST take fast path and log VoT audit entry.
    easy_choices = ["Paris", "Lyon"]
    pipeline.sys1 = _stub_sys1("general", easy_choices)
    easy_res = pipeline.evaluate("Which city is the capital of France?", easy_choices, domain="general")
    assert easy_res.fast_path_taken is True
    assert easy_res.recommended_thinking == "off"
    assert easy_res.vot_expected_utility_gain is not None
    assert easy_res.vot_lambda_cost == 1000.0
    assert any("VoT Gate:" in line and "FAST_PATH" in line for line in easy_res.audit_trail)


# -----------------------------------------------------------------------------
# 3. Gate 2: /v1/route Contract & pi-system2 Compatibility
# -----------------------------------------------------------------------------


def test_v1_route_emits_recommended_thinking_and_safety_always_on():
    """POST /v1/route preserves all legacy fields and emits recommended_thinking + delta_U."""
    mock_router = MagicMock()

    def fake_route(prompt: str, fallback_track: str = "general") -> RouteDecision:
        if "rm -rf" in prompt:
            return RouteDecision(
                selected_track="safety",
                confidence=0.99,
                probabilities={"safety": 0.99, "general": 0.01},
                method="lexical",
                latency_ms=0.05,
            )
        if "sourdough" in prompt:
            return RouteDecision(
                selected_track="culinary",
                confidence=0.95,
                probabilities={"culinary": 0.95, "general": 0.05},
                method="lexical",
                latency_ms=0.05,
            )
        return RouteDecision(
            selected_track=fallback_track,
            confidence=0.12,
            probabilities={"general": 0.12, "science": 0.11, "code": 0.11},
            method="fallback",
            latency_ms=0.40,
        )

    mock_router.route.side_effect = fake_route
    app = create_app(engine=MagicMock(), router=mock_router, device="cpu")
    tc = TestClient(app)

    # Safety prompt ALWAYS emits recommended_thinking="on" even at lambda_cost=1000.0
    r_safe = tc.post("/v1/route", json={"prompt": "rm -rf /", "lambda_cost": 1000.0})
    assert r_safe.status_code == 200
    d_safe = r_safe.json()
    for key in ("track", "selected_track", "track_name", "confidence", "probabilities", "method", "latency_ms"):
        assert key in d_safe
    assert d_safe["track"] == "safety"
    assert d_safe["recommended_thinking"] == "on"
    assert d_safe["expected_utility_gain"] == 1.0

    # High-confidence culinary prompt turns thinking off at moderate lambda, on at tiny lambda
    r_easy_off = tc.post("/v1/route", json={"prompt": "Explain sourdough fermentation.", "lambda_cost": 0.5})
    assert r_easy_off.status_code == 200
    d_easy_off = r_easy_off.json()
    assert d_easy_off["track"] == "culinary"
    assert d_easy_off["recommended_thinking"] == "off"
    assert 0.0 <= d_easy_off["expected_utility_gain"] <= 1.0

    r_easy_on = tc.post("/v1/route", json={"prompt": "Explain sourdough fermentation.", "lambda_cost": 0.0})
    assert r_easy_on.status_code == 200
    assert r_easy_on.json()["recommended_thinking"] == "on"

    # Negative lambda_cost is rejected cleanly
    r_bad = tc.post("/v1/route", json={"prompt": "hello", "lambda_cost": -0.1})
    assert r_bad.status_code == 422


# -----------------------------------------------------------------------------
# 4. Primary + Goodhart: Held-Out Calibration Benchmark vs Static 0.70 / 0.25
# -----------------------------------------------------------------------------


def test_vot_gate_beats_static_pareto_frontier_on_held_out_split():
    """VoTGate reduces unnecessary escalations >= 30% at >= -0.5% accuracy and >= 85% hard recall."""
    train_eps, test_eps = generate_calibration_episodes(n_episodes=600, seed=42)

    gate = ValueOfThinkingGate(default_lambda_cost=0.35)
    fit_stats = gate.fit(train_eps)
    assert gate.is_fitted is True
    assert fit_stats["val_nll"] < 0.5
    assert fit_stats["temperature"] > 0.0

    static_metrics = evaluate_policy(test_eps, static_baseline_flags(test_eps))
    vot_flags = [gate.decide(ep.features, lambda_cost=0.35).escalate for ep in test_eps]
    vot_metrics = evaluate_policy(test_eps, vot_flags)

    unnecessary_reduction = (
        (static_metrics["unnecessary_escalations"] - vot_metrics["unnecessary_escalations"])
        / max(1.0, static_metrics["unnecessary_escalations"])
    )
    accuracy_delta = vot_metrics["accuracy"] - static_metrics["accuracy"]

    # Scorecard Primary: >= 30% reduction in unnecessary escalations at >= -0.5% accuracy delta
    assert unnecessary_reduction >= 0.30, (
        f"Expected >= 30% unnecessary escalation reduction, got {unnecessary_reduction:.1%}"
    )
    assert accuracy_delta >= -0.005, (
        f"Expected accuracy delta >= -0.5%, got {accuracy_delta:.2%}"
    )
    # Scorecard Goodhart guard: >= 85% hard-question escalation recall
    assert vot_metrics["hard_recall"] >= 0.85, (
        f"Expected hard-question recall >= 85%, got {vot_metrics['hard_recall']:.1%}"
    )

    # Monotonicity across lambda sweep: higher lambda_cost never increases escalation rate.
    lambdas = [0.05, 0.20, 0.35, 0.55, 0.80]
    esc_rates = [
        evaluate_policy(test_eps, [gate.decide(ep.features, lambda_cost=lam).escalate for ep in test_eps])[
            "escalation_rate"
        ]
        for lam in lambdas
    ]
    assert esc_rates == sorted(esc_rates, reverse=True), f"Non-monotonic escalation rates: {esc_rates}"


# -----------------------------------------------------------------------------
# 5. JSON Serialization Round-Trip (No Pickle)
# -----------------------------------------------------------------------------


def test_vot_gate_json_save_load_roundtrip(tmp_path: Path):
    """Saved gate is plain JSON and reloads with identical decision outputs."""
    train_eps, test_eps = generate_calibration_episodes(n_episodes=160, seed=13)
    gate = ValueOfThinkingGate(default_lambda_cost=0.40)
    gate.fit(train_eps)

    out_file = gate.save(tmp_path / "vot_gate.json")
    raw = json.loads(out_file.read_text(encoding="utf-8"))
    assert raw["schema_version"] == "1.0"
    assert len(raw["weights"]) == ValueOfThinkingGate.FEATURE_DIM

    loaded = ValueOfThinkingGate.load(out_file)
    for ep in test_eps[:25]:
        d1 = gate.decide(ep.features)
        d2 = loaded.decide(ep.features)
        assert d1 == d2


# -----------------------------------------------------------------------------
# 6. Q3 Latency Target: p95 <= 1.0ms over N=500 x 2 Runs
# -----------------------------------------------------------------------------


def test_vot_gate_decide_latency_under_1ms_p95():
    """ValueOfThinkingGate.decide() p95 latency stays well below 1.0ms over N=500 x 2 runs."""
    train_eps, test_eps = generate_calibration_episodes(n_episodes=600, seed=99)
    gate = ValueOfThinkingGate()
    gate.fit(train_eps)
    sample_features = [ep.features for ep in (test_eps + train_eps)[:500]]

    # Warmup
    for feat in sample_features[:20]:
        gate.decide(feat)

    for run_idx in range(2):
        latencies_ms: List[float] = []
        for feat in sample_features:
            t0 = time.perf_counter()
            gate.decide(feat)
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        p95 = float(np.percentile(latencies_ms, 95))
        assert p95 <= 1.0, f"Run {run_idx + 1} p95 latency {p95:.4f}ms exceeded 1.0ms budget"


# -----------------------------------------------------------------------------
# 7. AnyS1 Enumerable Skill Decision-Point Extractor
# -----------------------------------------------------------------------------


def test_extract_enumerable_decision_points_from_markdown():
    """Parses tables, arrow rules, If/When rules, and branch lists while skipping code blocks."""
    sample_skill = textwrap.dedent(
        """\
        ---
        name: demo-router
        description: Sample procedural skill for AnyS1 extraction.
        ---

        # Demo Router Skill

        ```markdown
        | Fake | Table |
        |---|---|
        | inside code | should be ignored |
        ```

        ## Routing Table

        | Input Type | Surface | Route |
        |---|---|---|
        | Bug report | Jira | skill `triage-bug` |
        | PR diff | GitHub | skill `adversarial-pr-review` |
        | Team update | Slack | skill `draft-message` |

        ## Execution Modes

        - **FAST_MODE**: Run single-pass System 1 classification.
        - **DEEP_MODE**: Escalate to full System 2 verification.

        ## Fallback Rules

        - High risk domain -> escalate to System 2 or human review
        - When confidence drops below 0.50: hand off to System 2 verifier.
        """
    )
    points = extract_enumerable_decision_points(sample_skill)
    assert len(points) >= 4
    kinds = {p.source_kind for p in points}
    assert {"table", "branch_list", "conditional"}.issubset(kinds)

    table_dp = next(p for p in points if p.source_kind == "table")
    assert len(table_dp.choices) == 3
    assert all("inside code" not in c for c in table_dp.choices)

    # Single-pass S1 resolution accepts high-confidence decision, falls back when uncertain
    gate = ValueOfThinkingGate(default_lambda_cost=0.35)
    chosen, dec_fast = resolve_decision_point_with_gate(
        table_dp,
        choice_probabilities=[0.93, 0.04, 0.03],
        gate=gate,
        domain="general",
        head_consensus_ratio=1.0,
    )
    assert chosen == table_dp.choices[0]
    assert dec_fast.escalate is False

    fallback_choice, dec_slow = resolve_decision_point_with_gate(
        table_dp,
        choice_probabilities=[0.35, 0.34, 0.31],
        gate=gate,
        domain="math",
        head_consensus_ratio=0.4,
    )
    assert fallback_choice is None
    assert dec_slow.escalate is True


def test_extract_enumerable_decision_points_on_real_skill_doc():
    """Verify clean extraction on a real SKILL.md when present on disk."""
    candidates = [
        Path.home() / "Brain" / ".agents" / "skills" / "work-update-router" / "SKILL.md",
        Path.home() / "pi-system2" / "packages" / "coding-agent" / "skills" / "system2-software-factory" / "SKILL.md",
    ]
    existing = [p for p in candidates if p.is_file()]
    if not existing:
        pytest.skip("No local SKILL.md files present in this environment")

    for skill_path in existing:
        points = extract_enumerable_decision_points(skill_path.read_text(encoding="utf-8"))
        assert len(points) >= 1, f"Expected >= 1 decision point in {skill_path.name}"
        assert all(isinstance(dp, SkillDecisionPoint) and len(dp.choices) >= 1 for dp in points)
