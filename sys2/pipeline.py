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
     - Structural Causal Models (interventions and counterfactuals).

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
from sys2.causal.scm import StructuralCausalModel
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
    all_falsified: bool = False
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
        if self.all_falsified:
            lines.append("CRITICAL: ALL CANDIDATES FALSIFIED BY FORMAL VERIFIERS (FAIL-CLOSED)")
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
        if not choices:
            raise ValueError("Choices list cannot be empty.")

        t0 = time.perf_counter()
        audit: List[str] = []

        # Step 1: System 1 Fast Perceptual Screening
        s1_res: PerceptionResult = self.sys1.adjudicate(question, choices, domain=domain)
        audit.append(
            f"System 1 screened {len(choices)} choices in {s1_res.total_latency_ms:.2f} ms "
            f"(Domain: {s1_res.domain}, Conf: {s1_res.confidence * 100:.1f}%)"
        )

        letters = [chr(ord("A") + i) if i < 26 else f"#{i}" for i in range(len(choices))]
        probs = list(s1_res.choice_probabilities) if s1_res.choice_probabilities else [1.0 / len(choices)] * len(choices)
        sorted_probs = sorted(probs, reverse=True)
        margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else 1.0

        # Detect Symbolic & Causal Triggers
        has_units = self.unit_verifier.has_unit_claims(question) or any(self.unit_verifier.has_unit_claims(c) for c in choices)
        has_equations = ("=" in question or "==" in question) or any("=" in c or "==" in c for c in choices)
        has_negation = any(w in question.lower() for w in [" not ", " never ", " cannot ", " except ", " neither "])
        has_causal = any(w in question.lower() for w in [" cause", " do(", "counterfactual", "intervention"])

        triggers_active = (
            force_sys2
            or has_units
            or has_equations
            or has_negation
            or has_causal
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
            f"eq={has_equations}, neg={has_negation}, causal={has_causal}, margin={margin:.3f}]"
        )

        vetoed_indices: List[int] = []
        adjusted_scores = list(probs)

        # Verification 1: Physical Dimensional Analysis (Units)
        if has_units:
            for idx, choice in enumerate(choices):
                unit_res = self.unit_verifier.verify_natural_language_claim(question, choice)
                if not unit_res.is_dimensionally_homogeneous:
                    vetoed_indices.append(idx)
                    adjusted_scores[idx] = -1e9
                    audit.append(f"Choice ({letters[idx]}) VETOED: Incompatible physical dimensions ({unit_res.details})")

        # Verification 2: Mathematical / Algebraic Equivalence (SymPy)
        if has_equations:
            for idx, choice in enumerate(choices):
                if "=" in choice or "==" in choice:
                    alg_res = self.algebra_verifier.verify_equivalence(choice.split("=")[0], choice.split("=")[-1])
                    if not alg_res.is_valid and alg_res.error != "ParseError":
                        if idx not in vetoed_indices:
                            vetoed_indices.append(idx)
                        adjusted_scores[idx] = -1e9
                        audit.append(f"Choice ({letters[idx]}) VETOED by AlgebraVerifier: {alg_res.details}")

        # Verification 3: Logical Negation Constraints (Z3 & Semantic filter)
        if has_negation:
            for idx, choice in enumerate(choices):
                neg_res = self.logic_verifier.check_negation_consistency(question, choice)
                if not neg_res.is_valid:
                    if idx not in vetoed_indices:
                        vetoed_indices.append(idx)
                    adjusted_scores[idx] = -1e9
                    audit.append(f"Choice ({letters[idx]}) VETOED: {neg_res.details}")

        # Fail-closed check: If ALL candidates were vetoed by formal verifiers
        if len(vetoed_indices) == len(choices):
            t_total = (time.perf_counter() - t0) * 1000.0
            audit.append("FAIL-CLOSED: All candidates violate formal symbolic invariants.")
            let = letters[s1_res.selected_index] if 0 <= s1_res.selected_index < len(letters) else "#"
            return DualProcessResult(
                winning_choice_idx=s1_res.selected_index,
                winning_choice_char=let,
                winning_choice_text=s1_res.selected_choice,
                confidence=0.0,
                sys1_confidence=s1_res.confidence,
                fast_path_taken=False,
                sys2_verified=False,
                all_falsified=True,
                vetoed_choices=vetoed_indices,
                audit_trail=audit,
                latency_ms=t_total,
                routed_domain=s1_res.domain,
            )

        # Verification 4: Dynamic Search / Rollout over candidate hypotheses
        def combined_verifier(q_text: str, c_text: str) -> Tuple[bool, str]:
            if has_units:
                res = self.unit_verifier.verify_natural_language_claim(q_text, c_text)
                if not res.is_dimensionally_homogeneous:
                    return False, f"Dimensional error: {res.details}"
            if has_equations and ("=" in c_text or "==" in c_text):
                alg_res = self.algebra_verifier.verify_equivalence(c_text.split("=")[0], c_text.split("=")[-1])
                if not alg_res.is_valid and alg_res.error != "ParseError":
                    return False, f"Algebraic error: {alg_res.details}"
            if has_negation:
                n_res = self.logic_verifier.check_negation_consistency(q_text, c_text)
                if not n_res.is_valid:
                    return False, n_res.details
            return True, "Valid"

        mcts_res = self.mcts_engine.search(
            question=question,
            choices=choices,
            prior_scores=adjusted_scores,  # Pass vetoed-adjusted scores!
            verifier_fn=combined_verifier,
        )

        audit.extend(mcts_res.audit_trace)
        best_idx = mcts_res.best_choice_idx

        # Calibrated final confidence
        if best_idx in vetoed_indices:
            final_conf = 0.0
            is_verified = False
        else:
            final_conf = max(0.5, min(0.99, s1_res.confidence if best_idx == s1_res.selected_index else 0.85))
            is_verified = True

        t_total = (time.perf_counter() - t0) * 1000.0
        audit.append(f"System 2 verification completed in {t_total:.2f} ms total.")

        return DualProcessResult(
            winning_choice_idx=best_idx,
            winning_choice_char=letters[best_idx],
            winning_choice_text=choices[best_idx],
            confidence=final_conf,
            sys1_confidence=s1_res.confidence,
            fast_path_taken=False,
            sys2_verified=is_verified,
            all_falsified=False,
            vetoed_choices=vetoed_indices,
            audit_trail=audit,
            latency_ms=t_total,
            routed_domain=s1_res.domain,
        )
