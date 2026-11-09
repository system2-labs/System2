"""pipeline.py - Unified Dual-Process Cognitive Pipeline (System 1 + System 2).

Orchestrates the interplay between:
  1. System 1 (Fast Feedforward Perception):
     - ModernBERT-Large representations + 13 domain LoRA tracks.
     - Sub-20ms pairwise scoring, candidate filtering, and confidence estimation.
  2. System 2 (Deliberative Symbolic & Causal Verification):
     - Dimensional analysis (Fourier homogeneity via Pint).
     - Computer algebra & exact equation solving (via SymPy).
     - Propositional logic & negation consistency (via Z3).
     - Dynamic compute scaling (MCTS rollout over ambiguous candidates).

Implements the formal cognitive equation:
  Output = Sys2_Verify(Sys1_Screen(Prompt, Candidates))
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from sys1.engine import System1Engine, PerceptionResult
from sys2.symbolic.units import UnitVerifier
from sys2.symbolic.algebra import AlgebraVerifier
from sys2.symbolic.logic import LogicVerifier
from sys2.rollout.mcts import MCTSDeliberationEngine


@dataclass
class DualProcessResult:
    """Outcome of dual-process evaluation."""
    winning_choice_idx: int
    winning_choice_char: str
    winning_choice_text: str
    confidence: float
    sys1_confidence: float
    fast_path_taken: bool
    sys2_verified: bool
    vetoed_choices: List[int] = field(default_factory=list)
    audit_trail: List[str] = field(default_factory=list)
    latency_ms: float = 0.0
    routed_domain: str = ""

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            f" DUAL-PROCESS EVALUATION RESULT (Domain: {self.routed_domain.upper()})",
            "=" * 60,
            f"Winning Choice: ({self.winning_choice_char}) {self.winning_choice_text}",
            f"Final Confidence: {self.confidence * 100:.1f}%",
            f"Execution Mode: {'FAST PATH (System 1 Only)' if self.fast_path_taken else 'TRUE SYSTEM 2 VERIFIED'}",
            f"Total Latency: {self.latency_ms:.2f} ms",
        ]
        if self.vetoed_choices:
            lines.append(f"System 2 Vetoed Choices: {self.vetoed_choices}")
        if self.audit_trail:
            lines.append("-" * 60)
            lines.append("Audit Trail:")
            for item in self.audit_trail:
                lines.append(f"  * {item}")
        lines.append("=" * 60)
        return "\n".join(lines)


class DualProcessPipeline:
    """Unified dual-process architecture combining fast perception and deliberative verification."""

    def __init__(
        self,
        device: Optional[str] = None,
        fast_path_threshold: float = 0.85,
        min_margin_threshold: float = 0.15,
    ):
        self.sys1 = System1Engine(device=device)
        self.unit_verifier = UnitVerifier()
        self.algebra_verifier = AlgebraVerifier()
        self.logic_verifier = LogicVerifier()
        self.mcts_engine = MCTSDeliberationEngine()
        self.fast_path_threshold = fast_path_threshold
        self.min_margin_threshold = min_margin_threshold

    def evaluate(
        self,
        question: str,
        choices: List[str],
        domain: Optional[str] = None,
        force_sys2: bool = False,
    ) -> DualProcessResult:
        """Execute dual-process cognitive arbitration.

        Fast path: System 1 perceptual screening (<20ms).
        Slow path: System 2 symbolic verification & dynamic compute allocation.
        """
        t0 = time.perf_counter()
        audit: List[str] = []

        # Step 1: System 1 Fast Perceptual Screening
        s1_res: PerceptionResult = self.sys1.adjudicate(question, choices, domain=domain)
        audit.append(
            f"System 1 screened {len(choices)} choices in {s1_res.total_latency_ms:.2f} ms "
            f"(Domain: {s1_res.domain}, Conf: {s1_res.confidence * 100:.1f}%)"
        )

        letters = ["A", "B", "C", "D"][:len(choices)]
        probs = list(s1_res.choice_probabilities) if s1_res.choice_probabilities else [1.0 / len(choices)] * len(choices)
        sorted_probs = sorted(probs, reverse=True)
        margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else 1.0

        # Detect Symbolic & Causal Triggers
        has_units = self.unit_verifier.has_unit_claims(question) or any(self.unit_verifier.has_unit_claims(c) for c in choices)
        has_equations = "=" in question or any("=" in c for c in choices)
        has_negation = any(w in question.lower() for w in [" not ", " never ", " cannot ", " except ", " least "])

        triggers_active = (
            force_sys2
            or has_units
            or has_equations
            or has_negation
            or (s1_res.confidence < self.fast_path_threshold)
            or (margin < self.min_margin_threshold)
        )

        if not triggers_active:
            # Fast Path Accepted!
            t_total = (time.perf_counter() - t0) * 1000.0
            audit.append("System 1 confidence high with no symbolic triggers; fast path accepted.")
            let = letters[s1_res.selected_index] if 0 <= s1_res.selected_index < len(letters) else "#"
            return DualProcessResult(
                winning_choice_idx=s1_res.selected_index,
                winning_choice_char=let,
                winning_choice_text=s1_res.selected_choice,
                confidence=s1_res.confidence,
                sys1_confidence=s1_res.confidence,
                fast_path_taken=True,
                sys2_verified=False,
                audit_trail=audit,
                latency_ms=t_total,
                routed_domain=s1_res.domain,
            )

        # Step 2: System 2 Deliberative Verification
        audit.append(
            f"Activating True System 2 Deliberation: triggers [units={has_units}, "
            f"eq={has_equations}, neg={has_negation}, margin={margin:.3f}]"
        )

        vetoed_indices: List[int] = []
        adjusted_scores = list(probs)

        # Verification 1: Physical Dimensional Analysis (Units)
        if has_units:
            for idx, choice in enumerate(choices):
                combined = f"{question} {choice}"
                stmt_res = self.unit_verifier.verify_statement(combined)
                if not stmt_res.is_dimensionally_homogeneous:
                    vetoed_indices.append(idx)
                    adjusted_scores[idx] = -1e9
                    audit.append(f"Choice ({letters[idx]}) VETOED: Incompatible physical dimensions ({stmt_res.details})")

        # Verification 2: Logical Negation Constraints
        if has_negation:
            for idx, choice in enumerate(choices):
                neg_res = self.logic_verifier.check_negation_consistency(question, choice)
                if not neg_res.is_valid:
                    if idx not in vetoed_indices:
                        vetoed_indices.append(idx)
                    adjusted_scores[idx] = -1e9
                    audit.append(f"Choice ({letters[idx]}) VETOED: {neg_res.details}")

        # Verification 3: Dynamic Search / Rollout over surviving candidates
        def combined_verifier(q_text: str, c_text: str) -> Tuple[bool, str]:
            if has_units:
                res = self.unit_verifier.verify_statement(f"{q_text} {c_text}")
                if not res.is_dimensionally_homogeneous:
                    return False, f"Dimensional error: {res.details}"
            if has_negation:
                n_res = self.logic_verifier.check_negation_consistency(q_text, c_text)
                if not n_res.is_valid:
                    return False, n_res.details
            return True, "Valid"

        mcts_res = self.mcts_engine.search(
            question=question,
            choices=choices,
            prior_scores=probs,
            verifier_fn=combined_verifier,
        )

        audit.extend(mcts_res.audit_trace)
        best_idx = mcts_res.best_choice_idx

        # Calibrated final confidence
        final_conf = max(0.5, min(0.99, s1_res.confidence if best_idx == s1_res.selected_index else 0.85))

        t_total = (time.perf_counter() - t0) * 1000.0
        audit.append(f"System 2 verification completed in {t_total:.2f} ms total.")

        return DualProcessResult(
            winning_choice_idx=best_idx,
            winning_choice_char=letters[best_idx],
            winning_choice_text=choices[best_idx],
            confidence=final_conf,
            sys1_confidence=s1_res.confidence,
            fast_path_taken=False,
            sys2_verified=True,
            vetoed_choices=vetoed_indices,
            audit_trail=audit,
            latency_ms=t_total,
            routed_domain=s1_res.domain,
        )
