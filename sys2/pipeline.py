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

import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

try:
    import sympy
    from sympy.parsing.sympy_parser import (
        convert_xor,
        parse_expr,
        standard_transformations,
    )

    _SYMPY_AVAILABLE = True
except ImportError:
    sympy = None
    _SYMPY_AVAILABLE = False

from routing.vot_gate import ValueOfThinkingGate, VoTDecision, VoTFeatures
from sys1.engine import System1Engine, PerceptionResult
from sys2.parser import (
    SemanticParser,
    AlgebraicClaim,
    CausalClaim,
)
from sys2.symbolic.units import UnitVerifier
from sys2.symbolic.algebra import AlgebraVerifier
from sys2.symbolic.logic import LogicVerifier
from sys2.causal.scm import CausalQueryResult, StructuralCausalModel
from sys2.rollout.mcts import MCTSDeliberationEngine


@dataclass
class _CausalVerificationContext:
    """Compiled StructuralCausalModel state for a single evaluation prompt."""

    scm: StructuralCausalModel
    edges: List[Tuple[str, str]]
    interventions: Dict[str, Any]
    factual_obs: Dict[str, Any]
    is_counterfactual: bool
    target_var: Optional[str]
    query_result: CausalQueryResult
    expected_state: Dict[str, Any]
    upstream_ancestors: Dict[str, Set[str]]
    var_lookup: Dict[str, str]


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
    vot_expected_utility_gain: Optional[float] = None
    vot_lambda_cost: Optional[float] = None
    recommended_thinking: Optional[str] = None

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

    # Strict mathematical token grammar for _extract_equation: matches algebraic atoms
    # joined only by explicit operators or adjacent parentheses so surrounding English prose
    # words (e.g. "Solve the equation ... for x") are never swallowed into the equation.
    _MATH_FUNCS = r"(?:sin|cos|tan|log|ln|exp|sqrt|abs)"
    _MATH_ATOM = (
        r"(?:"
        + _MATH_FUNCS
        + r"\s*\([^()]+\)(?:\s*(?:\^|\*\*)\s*[+-]?\d+)?"
        r"|(?:\d+(?:\.\d+)?)?\([^()]+\)(?:\s*(?:\^|\*\*)\s*[+-]?\d+)?"
        r"|\d+(?:\.\d+)?[a-zA-Z](?:\s*(?:\^|\*\*)\s*[+-]?\d+)?"
        r"|\d+(?:\.\d+)?"
        r"|[a-zA-Z](?:\s*(?:\^|\*\*)\s*[+-]?\d+)?"
        r")"
    )
    _MATH_SIGNED_ATOM = r"[+-]?\s*" + _MATH_ATOM
    _MATH_EXPR = (
        _MATH_SIGNED_ATOM
        + r"(?:\s*(?:\*\*|[+\-*/^])\s*"
        + _MATH_SIGNED_ATOM
        + r"|\s*\([^()]+\)(?:\s*(?:\^|\*\*)\s*[+-]?\d+)?)*"
    )
    _EQUATION_RE = re.compile(
        r"(?<![A-Za-z0-9_])("
        + _MATH_EXPR
        + r")\s*(?<![!<>:=])(==?)(?![=>])\s*("
        + _MATH_EXPR
        + r")(?![A-Za-z0-9_])"
    )
    _UNIT_AFTER_NUM_RE = re.compile(
        r"^\s*(?:m/s\^?2?|m/s²|km/h|mph|kg|mg|cm|mm|km|Hz|Pa|kPa|N|J|W|V|A|K|m|s|h|g|L|Ω|"
        r"meters?|metres?|seconds?|kilograms?|grams?|newtons?|joules?|watts?|volts?|"
        r"amperes?|ohms?|pascals?|kelvins?|hours?|minutes?)\b",
        re.IGNORECASE,
    )
    _DO_CALL_RE = re.compile(r"\bdo\s*\([^)]*\)", re.IGNORECASE)

    # Stop words that must never become causal DAG node names from natural-language regexes.
    _CAUSAL_STOP_WORDS = {
        "a", "an", "the", "it", "its", "this", "that", "these", "those",
        "what", "which", "who", "whom", "whose", "why", "how", "where", "when",
        "if", "then", "else", "and", "or", "not", "no", "yes", "nor", "but",
        "is", "are", "was", "were", "be", "been", "being", "do", "does", "did",
        "has", "have", "had", "can", "could", "will", "would", "shall", "should",
        "may", "might", "must", "in", "on", "at", "to", "for", "of", "by", "with",
        "from", "under", "over", "into", "onto", "as", "so", "than", "there", "here",
        "we", "you", "they", "he", "she", "i", "one", "some", "any", "all", "each",
        "both", "fact", "factual", "true", "false", "value", "result", "statement",
        "option", "choice", "model", "graph", "causal", "dag", "scm", "variable",
        "node", "system", "event", "factor", "something", "nothing", "change",
        "changes", "effect", "cause", "causes", "intervention", "counterfactual",
    }

    # Regex capturing "Var = <rhs>" segments without greedily crossing clause boundaries
    # so chained equations ("Y = 2*X and Z = Y + 3" or "Y = 2*X and X = 1") all match.
    # Uses \.(?!\d) so decimal literals like 2.5*X or 1.5 are not truncated at the dot.
    _SCM_ASSIGN_RE = re.compile(
        r"(?<![A-Za-z0-9_])([A-Za-z_][A-Za-z0-9_]*)\s*(?<![!<>:=])=(?![=>])\s*"
        r"(.+?)(?=\s*(?:[,;?!\n]|\.(?!\d)|$|\b(?:and|with|where|when|what|under|given|"
        r"then|for|so|but|while|because|assuming|factually|suppose)\b))",
        re.IGNORECASE,
    )

    def __init__(
        self,
        device: Optional[str] = None,
        fast_path_threshold: float = 0.70,
        min_margin_threshold: float = 0.25,
        vot_gate: Optional[ValueOfThinkingGate] = None,
        lambda_cost: Optional[float] = None,
    ):
        self.sys1 = System1Engine(device=device)
        self.parser = SemanticParser()
        self.unit_verifier = UnitVerifier()
        self.algebra_verifier = AlgebraVerifier()
        self.logic_verifier = LogicVerifier()
        self.mcts_engine = MCTSDeliberationEngine()
        self.fast_path_threshold = fast_path_threshold
        self.min_margin_threshold = min_margin_threshold
        self.vot_gate = vot_gate
        self.lambda_cost = lambda_cost

    def _substitute_physics_vars(self, question: str, choice: str) -> str:
        """Normalize compound units and map symbolic physics variables to SI units.

        Shared by Step 2 dimensional verification and MCTS combined_verifier so both
        evaluate symbolic expressions (e.g. '0.5 * m * v^2' vs 'm * v') identically.
        Order matters: 'm -> kg' runs before 'v -> (m/s)' and 'a -> (m/s^2)' so the
        meter token introduced inside velocity/acceleration is never rewritten to kg.
        """
        subbed_choice = self.parser.normalize_units(choice)
        if any(
            k in question.lower()
            for k in (
                "mass",
                "velocity",
                "speed",
                "acceleration",
                "force",
                "energy",
                "work",
                "momentum",
                "power",
                "pressure",
            )
        ):
            # Preserve explicit unit occurrences like "5 m" or "m/s" via lookarounds.
            physics_var_subs = (
                (r"(?<!\d\s)(?<!\d)\bm\b(?!\s*/\s*s)", "kg"),
                (r"(?<!\d\s)(?<!\d)\bv\b", "(m/s)"),
                (r"(?<!\d\s)(?<!\d)\ba\b", "(m/s^2)"),
                (r"(?<!\d\s)(?<!\d)\bt\b", "s"),
                (r"(?<!\d\s)(?<!\d)\bF\b", "N"),
            )
            for pat, unit_rep in physics_var_subs:
                subbed_choice = re.sub(pat, unit_rep, subbed_choice)
        return subbed_choice

    def _extract_equation(self, text: str) -> Optional[str]:
        """Extract a genuine mathematical equation from text without swallowing English prose.

        Masks Pearl do(...) calls and rejects physical unit assignments (e.g. 'Work = 10 N * 5 m',
        'm = 5 kg') and prose sentences containing '='.
        """
        if not text or "=" not in text:
            return None

        cleaned = self._DO_CALL_RE.sub(lambda m: " " * len(m.group(0)), text)
        target_m = re.search(
            r"\b(?:solve\s+for|for|find|value\s+of)\s+([a-zA-Z])\b",
            cleaned,
            re.IGNORECASE,
        )
        target_var = target_m.group(1) if target_m else None

        candidates: List[Tuple[str, str, str, str]] = []
        allowed_funcs = {"sin", "cos", "tan", "log", "ln", "exp", "sqrt", "abs"}

        for m in self._EQUATION_RE.finditer(cleaned):
            lhs = m.group(1).strip()
            op = m.group(2)
            rhs = m.group(3).strip()
            full = f"{lhs} {op} {rhs}"

            # Reject when RHS ends with a number immediately followed by a physical unit
            # (e.g. "W = 10 N * 5 m" or "m = 5 kg").
            tail = cleaned[m.end() :]
            if re.search(r"\d$", rhs) and self._UNIT_AFTER_NUM_RE.match(tail):
                continue
            if self.unit_verifier.has_unit_claims(full):
                continue

            # Require at least one digit or mathematical operator so prose like "I = a" is ignored.
            if not re.search(r"[\d+*^/()-]", lhs + rhs):
                continue
            if full.count("(") != full.count(")"):
                continue

            # Reject any multi-letter English prose words hidden inside parenthesized groups.
            inner_words = re.findall(r"[A-Za-z]{2,}", full)
            if any(w.lower() not in allowed_funcs for w in inner_words):
                continue

            candidates.append((lhs, op, rhs, full))

        if not candidates:
            return None

        if target_var:
            for lhs, _op, rhs, full in candidates:
                if re.search(rf"\b{re.escape(target_var)}\b", f"{lhs} {rhs}"):
                    return full

        candidates.sort(key=lambda item: len(item[3]), reverse=True)
        return candidates[0][3]

    def _is_algebraic_identity_candidate(self, choice: str) -> Optional[Tuple[str, str]]:
        """Return (lhs, rhs) iff `choice` is a pure algebraic identity or numerical equality.

        Prevents unit-bearing physics strings ('Work = 10 N * 5 m'), prose sentences,
        function definitions ('f(x) = x^2'), and single-variable assignments ('x = 3',
        'Light = 1') from being falsely checked as universal algebraic identities.
        """
        if not choice or "=" not in choice:
            return None
        if self.unit_verifier.has_unit_claims(choice):
            return None

        extracted = self._extract_equation(choice)
        if not extracted:
            return None

        # The extracted equation must span the entire cleaned choice; prose with an incidental
        # '=' (e.g. "Because x = 5, the loop exits") is not an algebraic identity claim.
        norm_choice = re.sub(r"\s+", "", choice.strip().strip("$").rstrip(".;").strip())
        norm_extracted = re.sub(r"\s+", "", extracted)
        if norm_choice != norm_extracted:
            return None

        op = "==" if "==" in extracted else "="
        lhs, rhs = [p.strip() for p in extracted.split(op, 1)]
        if self.parser.resolve_target_concept(lhs) is not None:
            return None

        # Skip function definitions such as "f(x) = x^2" or "g(t) = 2*t".
        if re.match(r"^[a-zA-Z]\s*\([a-zA-Z,\s]+\)$", lhs):
            return None

        # Skip fresh variable bindings ("y = 2*x + 5", "x = 3") where a single LHS symbol
        # does not appear on RHS; note "x = x + 1" still passes because x appears on both sides.
        if re.match(r"^[a-zA-Z]$", lhs) and not re.search(rf"\b{re.escape(lhs)}\b", rhs):
            return None

        allowed_funcs = {"sin", "cos", "tan", "log", "ln", "exp", "sqrt", "abs"}
        lhs_vars = {w for w in re.findall(r"[A-Za-z]+", lhs) if w.lower() not in allowed_funcs}
        rhs_vars = {w for w in re.findall(r"[A-Za-z]+", rhs) if w.lower() not in allowed_funcs}
        has_func = any(w.lower() in allowed_funcs for w in re.findall(r"[A-Za-z]{2,}", lhs + " " + rhs))

        # Verify as identity only when using explicit '==', pure numeric arithmetic,
        # shared symbolic variables across both sides, or math function identities.
        if op == "==" or (not lhs_vars and not rhs_vars) or bool(lhs_vars & rhs_vars) or has_func:
            return lhs, rhs

        return None

    def _verify_algebraic_choice(
        self,
        question: str,
        q_eq: Optional[str],
        target_var: Optional[str],
        choice: str,
        alg_claim: Optional[AlgebraicClaim] = None,
    ) -> Tuple[str, str]:
        """Verify a candidate choice algebraically (equation solving + identity check).

        Returns (verdict, detail) where verdict is 'confirm', 'veto', or 'neutral'.
        Shared between Step 2 and MCTS combined_verifier.
        """
        # 1. Equation solving: question states an equation (e.g. "2*x + 4 = 10") and choice
        #    supplies either "x = 3" or a bare numeric/symbolic answer when target_var is named.
        if q_eq and not self.unit_verifier.has_unit_claims(choice):
            stripped = choice.strip()
            var: Optional[str] = None
            val: Optional[str] = None

            c_assign = re.match(r"^([a-zA-Z])\s*=\s*([+-]?\s*[a-zA-Z0-9_./()-]+)$", stripped)
            if c_assign:
                var = c_assign.group(1)
                val = c_assign.group(2).strip()
            elif target_var and re.fullmatch(r"[+-]?\s*\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?", stripped):
                var = target_var
                val = stripped

            if var and val:
                sol_res = self.algebra_verifier.solve_for(q_eq, var)
                if sol_res.solutions:
                    # Do not falsely veto a concrete numeric candidate against an unresolved
                    # symbolic solution (e.g. solving "Y = 2*X" for Y yields ["2*X"]).
                    val_expr = self.algebra_verifier.parse(val)
                    val_has_syms = bool(getattr(val_expr, "free_symbols", set()))
                    sol_has_unresolved = False
                    if not val_has_syms:
                        sol_exprs = [self.algebra_verifier.parse(s) for s in sol_res.solutions]
                        if sol_exprs and all(
                            e is not None and bool(getattr(e, "free_symbols", set())) for e in sol_exprs
                        ):
                            sol_has_unresolved = True

                    if not sol_has_unresolved:
                        if val in sol_res.solutions or any(
                            self.algebra_verifier.verify_equivalence(val, s).is_valid for s in sol_res.solutions
                        ):
                            return "confirm", f"exact solution for {q_eq}: {var} = {val}"
                        return "veto", f"{var} = {val} is not a valid solution for {q_eq}"

        # 2. Algebraic identity / numerical equality check (tightened so physics/prose/bindings
        #    are never falsely routed to AlgebraVerifier).
        identity_pair = self._is_algebraic_identity_candidate(choice)
        if identity_pair is None and alg_claim is not None and "=" in choice:
            # Allow explicit '==' claims only when choice passes unit/prose/binding guards.
            identity_pair = self._is_algebraic_identity_candidate(f"{alg_claim.lhs} {alg_claim.relation} {alg_claim.rhs}")

        if identity_pair is not None:
            lhs, rhs = identity_pair
            alg_res = self.algebra_verifier.verify_equivalence(lhs, rhs)
            if not alg_res.is_valid and alg_res.error != "ParseError":
                return "veto", f"Algebraic error: {alg_res.details}"

        return "neutral", "Valid"

    @classmethod
    def _is_causal_stop_word(cls, token: str, formal_syntax: bool = False) -> bool:
        """Return True if token is an English stop word rather than a causal variable name.

        In formal DAG syntax ('A -> B -> C', 'do(A = 1)', 'B = 2*A'), uppercase single-letter
        symbols like 'A' and 'I' are legitimate mathematical variable names rather than
        English articles/pronouns.
        """
        if formal_syntax and len(token) == 1 and token.isupper():
            return False
        return token.lower() in cls._CAUSAL_STOP_WORDS

    @staticmethod
    def _parse_causal_literal(raw: str) -> Any:
        """Parse a literal value from causal text into int, float, bool, or str."""
        s = raw.strip().rstrip(".,;:")
        lower = s.lower()
        if lower == "true":
            return True
        if lower == "false":
            return False
        try:
            return int(s)
        except ValueError:
            pass
        try:
            return float(s)
        except ValueError:
            return s

    @staticmethod
    def _is_causal_literal(raw: str) -> bool:
        """Return True if raw string is a pure numeric or boolean literal."""
        return bool(re.fullmatch(r"[+-]?\d+(?:\.\d+)?|true|false", raw.strip(), re.IGNORECASE))

    @staticmethod
    def _causal_values_equal(a: Any, b: Any) -> bool:
        """Compare two SCM variable values with float tolerance and bool/int compatibility."""
        if isinstance(a, (int, float, bool)) and isinstance(b, (int, float, bool)):
            return abs(float(a) - float(b)) < 1e-6
        return str(a).strip().lower() == str(b).strip().lower()

    @classmethod
    def _parse_structural_equation(cls, rhs: str) -> Optional[Tuple[Callable[..., Any], List[str]]]:
        """Safely compile a structural equation RHS into a Python callable without eval().

        Supports arithmetic expressions (e.g. '2*X', '2X + 3', 'X + Y', 'Switch', '1 - Wire'),
        boolean negation ('not Wire'), and simple binary ternary ('1 if Wire == 1 else 0').
        """
        cleaned = rhs.strip()
        if not cleaned:
            return None

        # 1. Safe ternary form: "1 if Wire == 1 else 0"
        ternary_m = re.fullmatch(
            r"([+-]?\d+(?:\.\d+)?)\s+if\s+([A-Za-z_][A-Za-z0-9_]*)\s*==\s*"
            r"([+-]?\d+(?:\.\d+)?)\s+else\s+([+-]?\d+(?:\.\d+)?)",
            cleaned,
            re.IGNORECASE,
        )
        if ternary_m:
            v_true = cls._parse_causal_literal(ternary_m.group(1))
            p_var = ternary_m.group(2)
            cmp_val = cls._parse_causal_literal(ternary_m.group(3))
            v_false = cls._parse_causal_literal(ternary_m.group(4))
            if p_var.lower() in cls._CAUSAL_STOP_WORDS:
                return None

            def _ternary_eq(**kwargs: Any) -> Any:
                pv = kwargs.get(p_var)
                if pv is None:
                    pv = next((v for k, v in kwargs.items() if k.lower() == p_var.lower()), None)
                if pv is None:
                    return None
                return v_true if cls._causal_values_equal(pv, cmp_val) else v_false

            return _ternary_eq, [p_var]

        # Trim trailing conditional clause ("2*X if we apply ...") if not a ternary.
        cleaned = re.split(r"\bif\b", cleaned, maxsplit=1, flags=re.IGNORECASE)[0].strip()
        if not cleaned:
            return None

        # 2. Simple boolean negation: "not Wire" or "!Wire"
        neg_m = re.fullmatch(r"(?:not\s+|!\s*)([A-Za-z_][A-Za-z0-9_]*)", cleaned, re.IGNORECASE)
        if neg_m:
            p_var = neg_m.group(1)
            if p_var.lower() in cls._CAUSAL_STOP_WORDS:
                return None

            def _neg_eq(**kwargs: Any) -> Any:
                pv = kwargs.get(p_var)
                if pv is None:
                    pv = next((v for k, v in kwargs.items() if k.lower() == p_var.lower()), None)
                if pv is None:
                    return None
                if isinstance(pv, bool):
                    return not pv
                return 0 if pv else 1

            return _neg_eq, [p_var]

        # 3. Arithmetic structural equation parsed via SymPy with explicit local_dict so
        #    multi-letter identifiers ('Switch', 'Wire') and built-in letters ('E', 'N', 'S')
        #    are treated as atomic symbols rather than split or evaluated as SymPy constants.
        if not _SYMPY_AVAILABLE or sympy is None:
            return None
        if not re.fullmatch(r"[A-Za-z0-9_+\-*/^().\s]+", cleaned):
            return None
        # Reject function calls and dunder/underscore-prefixed tokens for safety.
        if re.search(r"[A-Za-z_]\s*\(", cleaned):
            return None

        # Expand numeric implicit multiplication ("2X" -> "2*X") without splitting multi-letter names.
        prepped = re.sub(r"(\d(?:\.\d+)?)\s*([A-Za-z_][A-Za-z0-9_]*)", r"\1*\2", cleaned)
        tokens = re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", prepped)
        if not tokens:
            return None
        if any(t.startswith("_") or cls._is_causal_stop_word(t, formal_syntax=True) for t in tokens):
            return None

        local_dict = {t: sympy.Symbol(t) for t in set(tokens)}
        try:
            transforms = standard_transformations + (convert_xor,)
            expr = parse_expr(prepped, local_dict=local_dict, transformations=transforms, evaluate=True)
        except Exception:
            return None

        parents = sorted(str(s) for s in expr.free_symbols)
        if not parents:
            return None

        def _sympy_eq(**kwargs: Any) -> Any:
            # Fall back to case-insensitive lookup so mixed-case DAG/equation names (e.g. X -> Y with Y = 2*x) still bind.
            ci_kwargs = {k.lower(): v for k, v in kwargs.items()}
            resolved_vals: Dict[str, Any] = {}
            for p in parents:
                val_p = kwargs.get(p)
                if val_p is None:
                    val_p = ci_kwargs.get(p.lower())
                if val_p is None:
                    return None
                resolved_vals[p] = int(val_p) if isinstance(val_p, bool) else val_p
            subs_map = {sympy.Symbol(p): resolved_vals[p] for p in parents}
            val = sympy.simplify(expr.subs(subs_map))
            if getattr(val, "is_number", False):
                fval = float(val.evalf())
                return int(round(fval)) if abs(fval - round(fval)) < 1e-9 else fval
            return None

        return _sympy_eq, parents

    def _build_causal_context(
        self,
        question: str,
        q_causal: Optional[CausalClaim],
        causal_claims: List[CausalClaim],
    ) -> Optional[_CausalVerificationContext]:
        """Build and simulate a StructuralCausalModel from a formal causal prompt.

        Returns None (fail-open) on purely conceptual natural-language causal questions
        that do not define a formal DAG/structural model with an intervention or counterfactual.
        """
        # 1. Extract Pearl interventions and mask their spans before scanning for factual values
        #    so do(Wire = 1) or "if Switch had been 1" is never misread as a factual observation.
        interventions: Dict[str, Any] = {}
        masked_chars = list(question)

        def _mask_span(start: int, end: int) -> None:
            for i in range(start, end):
                masked_chars[i] = " "

        for m in self._DO_CALL_RE.finditer(question):
            _mask_span(m.start(), m.end())
            inner = m.group(0)[m.group(0).find("(") + 1 : -1]
            for assign in inner.split(","):
                am = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+?)\s*$", assign)
                if am and not self._is_causal_stop_word(am.group(1), formal_syntax=True):
                    interventions[am.group(1)] = self._parse_causal_literal(am.group(2))

        for pat in (
            r"\b(?:setting|set|intervene\s+on)\s+([A-Za-z_][A-Za-z0-9_]*)\s+(?:to|=)\s*([+-]?\d+(?:\.\d+)?|true|false|[A-Za-z0-9_.-]+)",
            r"\bif\s+([A-Za-z_][A-Za-z0-9_]*)\s+(?:had\s+been|were|was|is)\s+(?:set\s+to\s+)?([+-]?\d+(?:\.\d+)?|true|false|[A-Za-z0-9_.-]+)",
        ):
            for m in re.finditer(pat, question, re.IGNORECASE):
                var = m.group(1).strip()
                if self._is_causal_stop_word(var, formal_syntax=(len(var) == 1 and var.isupper())):
                    continue
                _mask_span(m.start(), m.end())
                interventions.setdefault(var, self._parse_causal_literal(m.group(2)))

        if q_causal and q_causal.interventions:
            for k, v in q_causal.interventions.items():
                if not self._is_causal_stop_word(k, formal_syntax=(len(k) == 1 and k.isupper())):
                    interventions.setdefault(k, self._parse_causal_literal(str(v)))

        if not interventions:
            return None

        non_interv_text = "".join(masked_chars)
        is_counterfactual = bool(
            (q_causal and q_causal.is_counterfactual)
            or any(c.is_counterfactual for c in causal_claims)
            or any(
                w in question.lower()
                for w in ("counterfactual", "would have been", "had been", "had not been", "what if")
            )
        )

        # 2. Extract directed DAG edges, explicit structural equations, and candidate factual obs.
        raw_edges: List[Tuple[str, str]] = []
        explicit_equations: Dict[str, Tuple[Callable[..., Any], List[str]]] = {}
        raw_factual_obs: Dict[str, Any] = {}

        # (a) Explicit arrow chains: A -> B -> C
        arrow_chains = re.findall(
            r"\b([A-Za-z_][A-Za-z0-9_]*(?:\s*(?:-->|->|→)\s*[A-Za-z_][A-Za-z0-9_]*)+)\b",
            question,
        )
        for chain in arrow_chains:
            nodes = [n.strip() for n in re.split(r"-->|->|→", chain) if n.strip()]
            for i in range(len(nodes) - 1):
                p, c = nodes[i], nodes[i + 1]
                if not self._is_causal_stop_word(p, formal_syntax=True) and not self._is_causal_stop_word(
                    c, formal_syntax=True
                ):
                    raw_edges.append((p, c))

        # (b) Explicit structural equations ("Y = 2*X", "Wire = Switch") and factual assignments ("Switch = 0").
        for m in self._SCM_ASSIGN_RE.finditer(non_interv_text):
            lhs = m.group(1).strip()
            rhs = m.group(2).strip()
            if self._is_causal_stop_word(lhs, formal_syntax=True):
                continue
            if self.unit_verifier.has_unit_claims(rhs):
                continue

            if self._is_causal_literal(rhs):
                raw_factual_obs[lhs] = self._parse_causal_literal(rhs)
            else:
                parsed_eq = self._parse_structural_equation(rhs)
                if parsed_eq is not None:
                    eq_fn, eq_parents = parsed_eq
                    if lhs not in eq_parents:
                        explicit_equations[lhs] = (eq_fn, eq_parents)
                        for p in eq_parents:
                            raw_edges.append((p, lhs))

        # (c) Verbal causal edges: "A causes B", "B because A", "Light was 0 because Switch was 0".
        for m in re.finditer(
            r"\b([A-Za-z_][A-Za-z0-9_]*)\s+(?:causes?|leads?\s+to|affects?|influences?|controls?|determines?)\s+"
            r"([A-Za-z_][A-Za-z0-9_]*)\b",
            question,
            re.IGNORECASE,
        ):
            p, c = m.group(1).strip(), m.group(2).strip()
            if not self._is_causal_stop_word(p, formal_syntax=True) and not self._is_causal_stop_word(
                c, formal_syntax=True
            ):
                raw_edges.append((p, c))

        for m in re.finditer(
            r"\b([A-Za-z_][A-Za-z0-9_]*)\s+(?:(?:is|was|=)\s+[A-Za-z0-9_.-]+\s+)?because\s+"
            r"([A-Za-z_][A-Za-z0-9_]*)\b",
            question,
            re.IGNORECASE,
        ):
            c, p = m.group(1).strip(), m.group(2).strip()
            if not self._is_causal_stop_word(p, formal_syntax=True) and not self._is_causal_stop_word(
                c, formal_syntax=True
            ):
                raw_edges.append((p, c))

        if not raw_edges:
            return None

        # 3. Canonicalize node names and filter out any cyclic edges so SCM stays a valid DAG.
        var_lookup: Dict[str, str] = {}
        ordered_nodes: List[str] = []

        def _canonical(name: str) -> str:
            low = name.lower()
            if low not in var_lookup:
                var_lookup[low] = name
                ordered_nodes.append(name)
            return var_lookup[low]

        edges: List[Tuple[str, str]] = []
        adj: Dict[str, Set[str]] = {}

        def _can_reach(src: str, dst: str) -> bool:
            stack = [src]
            seen: Set[str] = set()
            while stack:
                curr = stack.pop()
                if curr == dst:
                    return True
                if curr not in seen:
                    seen.add(curr)
                    stack.extend(adj.get(curr, ()))
            return False

        for p_raw, c_raw in raw_edges:
            p = _canonical(p_raw)
            c = _canonical(c_raw)
            if p == c or (p, c) in edges:
                continue
            if _can_reach(c, p):
                continue
            edges.append((p, c))
            adj.setdefault(p, set()).add(c)

        if not edges:
            return None

        # Map interventions to canonical DAG nodes; ignore if intervened var is outside the DAG.
        active_interventions: Dict[str, Any] = {}
        for k, v in interventions.items():
            if len(k) == 1:
                if k in ordered_nodes:
                    active_interventions[k] = v
            elif k.lower() in var_lookup:
                active_interventions[var_lookup[k.lower()]] = v

        if not active_interventions:
            return None

        # 4. Extract factual observations strictly for DAG nodes from non-interventional text.
        factual_obs: Dict[str, Any] = {}
        for k, v in raw_factual_obs.items():
            if len(k) == 1:
                if k in ordered_nodes:
                    factual_obs[k] = v
            elif k.lower() in var_lookup:
                factual_obs[var_lookup[k.lower()]] = v

        for node_name in ordered_nodes:
            v_pat = rf"\b{re.escape(node_name)}\b" if len(node_name) == 1 else rf"(?i:\b{re.escape(node_name)}\b)"
            for m in re.finditer(
                rf"{v_pat}\s+(?i:is|was|equals?|initially|starts?\s+at|started\s+at)\s*"
                r"([+-]?\d+(?:\.\d+)?|(?i:true|false))(?![A-Za-z0-9_])",
                non_interv_text,
            ):
                factual_obs.setdefault(node_name, self._parse_causal_literal(m.group(1)))

        # Canonicalize explicit equation keys and parent lists.
        canon_equations: Dict[str, Tuple[Callable[..., Any], List[str]]] = {}
        for lhs, (eq_fn, eq_parents) in explicit_equations.items():
            canon_lhs = var_lookup.get(lhs.lower(), lhs)
            canon_parents = [var_lookup.get(p.lower(), p) for p in eq_parents]
            canon_equations[canon_lhs] = (eq_fn, canon_parents)

        # 5. Assemble StructuralCausalModel.
        parents_by_node: Dict[str, List[str]] = {n: [] for n in ordered_nodes}
        for p, c in edges:
            if p not in parents_by_node[c]:
                parents_by_node[c].append(p)

        # Binary identity propagation ("Switch -> Wire -> Light" with 0/1 or bool values)
        # applies ONLY when all observed/intervened values are binary and no factual
        # observation contradicts identity propagation, so unknown non-binary mechanisms fail open.
        concrete_vals = list(factual_obs.values()) + list(active_interventions.values())
        is_binary_domain = bool(concrete_vals) and all(
            isinstance(v, (int, bool)) and v in (0, 1, True, False) for v in concrete_vals
        )

        def _identity_parent_eq(**kwargs: Any) -> Any:
            return next(iter(kwargs.values()))

        scm = StructuralCausalModel("PipelineSCM")
        for node_name in ordered_nodes:
            p_list = parents_by_node[node_name]
            is_exo = len(p_list) == 0
            default_val = factual_obs.get(node_name)
            eq_callable: Optional[Callable[..., Any]] = None

            if node_name in canon_equations:
                eq_callable = canon_equations[node_name][0]
            elif len(p_list) == 1 and is_binary_domain:
                parent_name = p_list[0]
                if (
                    parent_name in factual_obs
                    and node_name in factual_obs
                    and not self._causal_values_equal(factual_obs[parent_name], factual_obs[node_name])
                ):
                    eq_callable = None
                else:
                    eq_callable = _identity_parent_eq

            scm.add_variable(
                name=node_name,
                parents=p_list,
                equation=eq_callable,
                default_value=default_val,
                is_exogenous=is_exo,
            )

        # 6. Resolve query target variable (explicitly queried variable > downstream sink).
        target_var: Optional[str] = None
        for pat in (
            r"\bwhat\s+would\s+([A-Za-z_][A-Za-z0-9_]*)\s+have\s+been\b",
            r"\bwhat\s+(?:is|would|will|becomes?\s+of|happens?\s+to|occurs?\s+to)\s+(?:the\s+)?([A-Za-z_][A-Za-z0-9_]*)\b",
            r"\b(?:find|compute|calculate|determine|value\s+of)\s+([A-Za-z_][A-Za-z0-9_]*)\b",
        ):
            tm = re.search(pat, question, re.IGNORECASE)
            if tm:
                cand = tm.group(1)
                resolved = cand if (len(cand) == 1 and cand in ordered_nodes) else var_lookup.get(cand.lower())
                if resolved and resolved not in active_interventions:
                    target_var = resolved
                    break

        if target_var is None:
            sinks = [n for n in ordered_nodes if not adj.get(n)]
            desc_sinks = [
                s
                for s in sinks
                if s not in active_interventions and any(s in scm.descendants(iv) for iv in active_interventions)
            ]
            if desc_sinks:
                target_var = desc_sinks[-1]
            else:
                non_iv_sinks = [s for s in sinks if s not in active_interventions]
                if non_iv_sinks:
                    target_var = non_iv_sinks[-1]
                else:
                    non_iv = [n for n in ordered_nodes if n not in active_interventions]
                    target_var = non_iv[-1] if non_iv else ordered_nodes[-1]

        # 7. Run Pearl Layer 3 counterfactual or Layer 2 do-calculus intervention.
        try:
            if is_counterfactual and factual_obs:
                try:
                    query_res = scm.counterfactual(
                        factual_obs=factual_obs,
                        hypothetical_interventions=active_interventions,
                        target_var=target_var,
                    )
                except ValueError:
                    # Graceful fallback when an unrelated exogenous node is unobserved.
                    query_res = scm.do(
                        interventions=active_interventions,
                        target_var=target_var,
                        exogenous_values=factual_obs,
                    )
            else:
                query_res = scm.do(
                    interventions=active_interventions,
                    target_var=target_var,
                    exogenous_values=factual_obs,
                )
        except ValueError:
            return None

        expected_state = {k: v for k, v in query_res.state_trajectory.items() if v is not None}
        upstream_ancestors = {iv: scm.ancestors(iv) for iv in active_interventions}

        return _CausalVerificationContext(
            scm=scm,
            edges=edges,
            interventions=active_interventions,
            factual_obs=factual_obs,
            is_counterfactual=is_counterfactual,
            target_var=target_var,
            query_result=query_res,
            expected_state=expected_state,
            upstream_ancestors=upstream_ancestors,
            var_lookup=var_lookup,
        )

    def _verify_causal_choice(
        self,
        ctx: _CausalVerificationContext,
        choice: str,
    ) -> Tuple[str, str]:
        """Verify a candidate choice against Pearl's do-calculus and simulated SCM state.

        Returns ('confirm' | 'veto' | 'neutral', detail).
        Veto checks always run before confirmation so partially right choices that violate
        upstream edge-cutting (e.g. 'Light = 1 and Switch = 1') are strictly vetoed.
        """
        choice_no_do = self._DO_CALL_RE.sub(lambda m: " " * len(m.group(0)), choice)
        neg_guard_re = re.compile(
            r"\b(?:not|no|never|doesn't|does\s+not|do\s+not|didn't|cannot|can\s+not|can't|"
            r"won't|will\s+not|would\s+not|wouldn't|without|fails?\s+to|neither|unaffected|unchanged)\b",
            re.IGNORECASE,
        )
        clause_break_re = re.compile(
            r"[,.;!?]|\b(?:not|no|never|without|but|while|whereas|although|however|"
            r"instead|rather|leaving|unaffected|unchanged|doesn't|does\s+not|cannot|can't)\b",
            re.IGNORECASE,
        )
        change_verb_re = re.compile(
            r"\b(?:changes?|alters?|affects?|modifies?|flips?|updates?|forces?|"
            r"causes?|sets?|turns?|switches?|increases?|decreases?|drives?|triggers?)\b",
            re.IGNORECASE,
        )

        # 1. Pearl edge-cutting invariant: do(iv = v) severs incoming edges to iv, so no
        #    upstream ancestor of iv can change as a consequence of the intervention.
        for iv, ancestors in ctx.upstream_ancestors.items():
            iv_pat = rf"\b{re.escape(iv)}\b" if len(iv) == 1 else rf"(?i:\b{re.escape(iv)}\b)"
            for anc in ancestors:
                # Match single-letter variables (e.g. 'A') case-sensitively so lowercase
                # English article 'a' ("causes a change in C") never collides with node A.
                anc_pat = rf"\b{re.escape(anc)}\b" if len(anc) == 1 else rf"(?i:\b{re.escape(anc)}\b)"

                # (a) Reversed causal arrow from intervened node back to upstream ancestor.
                if re.search(rf"{iv_pat}\s*(?:-->|->|→)\s*{anc_pat}", choice_no_do):
                    return (
                        "veto",
                        f"Pearl edge-cutting violation: do({iv} = {ctx.interventions[iv]}) "
                        f"severs incoming edges and cannot create causal arrow {iv} -> {anc}.",
                    )

                # (b) Active change verb targeting upstream ancestor in the same affirmative clause.
                for vm in change_verb_re.finditer(choice_no_do):
                    pre_seg = choice_no_do[max(0, vm.start() - 45) : vm.start()]
                    # Trim pre_seg at last sentence/clause boundary.
                    last_punc = max(pre_seg.rfind(","), pre_seg.rfind("."), pre_seg.rfind(";"))
                    if last_punc != -1:
                        pre_seg = pre_seg[last_punc + 1 :]
                    if neg_guard_re.search(pre_seg):
                        continue

                    post_seg = choice_no_do[vm.end() : vm.end() + 50]
                    anc_m = re.search(anc_pat, post_seg)
                    if anc_m:
                        between = post_seg[: anc_m.start()]
                        after_anc = post_seg[anc_m.end() : anc_m.end() + 30]
                        if not clause_break_re.search(between) and not re.match(
                            r"\s*(?:is\s+not|does\s+not|remains?\s+unchanged|is\s+unaffected|stays?\s+unchanged)",
                            after_anc,
                            re.IGNORECASE,
                        ):
                            return (
                                "veto",
                                f"Pearl edge-cutting violation: do({iv} = {ctx.interventions[iv]}) "
                                f"severs incoming edges to {iv} and cannot alter upstream ancestor {anc}.",
                            )

                # (c) Upstream ancestor as subject of an affirmative change verb
                #     ("A changes", "A becomes 1", "A and C both change").
                subj_re = re.compile(
                    rf"{anc_pat}\s+(?i:(?:and\s+[A-Za-z_][A-Za-z0-9_]*\s+)?"
                    r"(?:both\s+|also\s+|then\s+|must\s+|will\s+|would\s+|necessarily\s+)?"
                    r"(?:changes?|is\s+changed|are\s+changed|becomes?\s+changed|is\s+altered|are\s+altered|alters?|"
                    r"is\s+affected|are\s+affected|flips?|is\s+flipped|is\s+updated|updates?|"
                    r"switches?\s+to|turns?\s+(?:on|off|into|to)|increases?|decreases?|"
                    r"is\s+forced|is\s+modified|is\s+set\s+to|"
                    r"becomes?\s+(?P<val>[+-]?\d+(?:\.\d+)?|true|false|on|off)))"
                )
                for sm in subj_re.finditer(choice_no_do):
                    pre_seg = choice_no_do[max(0, sm.start() - 35) : sm.start()]
                    last_punc = max(pre_seg.rfind(","), pre_seg.rfind("."), pre_seg.rfind(";"))
                    if last_punc != -1:
                        pre_seg = pre_seg[last_punc + 1 :]
                    if neg_guard_re.search(pre_seg):
                        continue
                    claimed_val_raw = sm.group("val")
                    if claimed_val_raw is not None and anc in ctx.expected_state:
                        if self._causal_values_equal(
                            self._parse_causal_literal(claimed_val_raw), ctx.expected_state[anc]
                        ):
                            continue
                    return (
                        "veto",
                        f"Pearl edge-cutting violation: do({iv} = {ctx.interventions[iv]}) "
                        f"severs incoming edges to {iv} and cannot alter upstream ancestor {anc}.",
                    )

        # 2. Quantitative / state-trajectory verification against simulated SCM state.
        assertions: List[Tuple[str, Any, bool]] = []
        for var in ctx.scm.nodes:
            v_pat = rf"\b{re.escape(var)}\b" if len(var) == 1 else rf"(?i:\b{re.escape(var)}\b)"
            neg_val_re = re.compile(
                rf"{v_pat}\s+(?i:(?:is|are|will\s+be|would\s+be|would\s+have\s+been|cannot\s+be)\s+not|"
                r"(?:does|do|will|would)\s+not\s+(?:become|equal|remain|stay|be)|!=)\s*"
                r"(?P<val>[+-]?\d+(?:\.\d+)?|(?i:true|false))(?![A-Za-z0-9_])"
            )
            neg_spans: List[Tuple[int, int]] = []
            for nm in neg_val_re.finditer(choice_no_do):
                neg_spans.append((nm.start(), nm.end()))
                assertions.append((var, self._parse_causal_literal(nm.group("val")), True))

            aff_val_re = re.compile(
                rf"{v_pat}\s*(?i:would\s+have\s+been|would\s+be|would\s+remain|"
                r"will\s+be|will\s+remain|becomes?|remains?|stays?|equals?|is\s+still|is|==?)\s*"
                r"(?P<val>[+-]?\d+(?:\.\d+)?|(?i:true|false))(?![A-Za-z0-9_])"
            )
            for am in aff_val_re.finditer(choice_no_do):
                if any(ns <= am.start() < ne for ns, ne in neg_spans):
                    continue
                assertions.append((var, self._parse_causal_literal(am.group("val")), False))

        # Support bare literal choices (e.g. "10" for "What is Y under do(X = 5)?").
        if not assertions and ctx.target_var and ctx.target_var in ctx.expected_state:
            stripped = choice.strip()
            if self._is_causal_literal(stripped):
                assertions.append((ctx.target_var, self._parse_causal_literal(stripped), False))

        matched_vars: Set[str] = set()
        for var, claimed_val, is_negated in assertions:
            if var not in ctx.expected_state:
                continue
            expected_val = ctx.expected_state[var]
            matches = self._causal_values_equal(claimed_val, expected_val)
            if not is_negated:
                if not matches:
                    return (
                        "veto",
                        f"SCM contradiction: choice asserts {var} = {claimed_val}, "
                        f"but SCM computes {var} = {expected_val} ({ctx.query_result.explanation})",
                    )
                matched_vars.add(var)
            else:
                if matches:
                    return (
                        "veto",
                        f"SCM contradiction: choice denies {var} = {expected_val}, "
                        f"contradicting SCM ({ctx.query_result.explanation})",
                    )

        non_iv_matches = matched_vars - set(ctx.interventions.keys())
        if (ctx.target_var and ctx.target_var in matched_vars) or non_iv_matches:
            return "confirm", f"SCM verified ({ctx.query_result.explanation})"

        return "neutral", "No formal SCM contradiction detected."

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

        # Semantic DSL Slot Extraction
        phys_target, phys_claims = self.parser.parse_physical_claims(question, choices)
        alg_claims = self.parser.parse_algebraic_claims(question, choices)
        log_claims = self.parser.parse_logical_claims(question, choices)
        q_causal, causal_claims = self.parser.parse_causal_claims(question, choices)

        # Detect Symbolic & Causal Triggers
        has_units = (
            (phys_target is not None)
            or any(bool(c.target_concept or c.extracted_quantities) for c in phys_claims)
            or self.unit_verifier.has_unit_claims(question)
            or any(self.unit_verifier.has_unit_claims(c) for c in choices)
        )
        has_equations = (
            any(c is not None for c in alg_claims)
            or ("=" in question or "==" in question)
            or any("=" in c or "==" in c for c in choices)
        )
        has_negation = (
            any(c.has_negation for c in log_claims)
            or any(w in question.lower() for w in [" not ", " never ", " cannot ", " except ", " neither "])
        )
        has_causal = (
            (q_causal is not None)
            or any(bool(c.causes or c.effects or c.interventions or c.is_counterfactual) for c in causal_claims)
            or any(w in question.lower() for w in [" cause", " do(", "counterfactual", "intervention"])
        )

        hard_triggers = force_sys2 or has_units or has_equations or has_negation or has_causal
        vot_decision: Optional[VoTDecision] = None

        if self.vot_gate is not None:
            # Gate 1 (Fail-Closed): safety track and symbolic/causal triggers ALWAYS
            # escalate to System 2 regardless of lambda_cost or VoT score.
            safety_trigger = (s1_res.domain or "").strip().lower() == "safety"
            hard_triggers_active = hard_triggers or safety_trigger
            vot_features = VoTFeatures.from_perception_result(
                s1_res,
                has_hard_trigger=hard_triggers_active,
            )
            vot_decision = self.vot_gate.decide(vot_features, lambda_cost=self.lambda_cost)
            triggers_active = hard_triggers_active or vot_decision.escalate
            audit.append(
                f"VoT Gate: delta_U={vot_decision.expected_utility_gain:.4f} vs "
                f"lambda={vot_decision.lambda_cost:.4f} -> "
                f"{'ESCALATE' if triggers_active else 'FAST_PATH'} "
                f"(thinking={vot_decision.recommended_thinking}, "
                f"hard_override={vot_decision.hard_trigger_override})"
            )
        else:
            triggers_active = (
                hard_triggers
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
                vot_expected_utility_gain=vot_decision.expected_utility_gain if vot_decision else None,
                vot_lambda_cost=vot_decision.lambda_cost if vot_decision else None,
                recommended_thinking=vot_decision.recommended_thinking if vot_decision else None,
            )

        # Step 2: System 2 Deliberative Verification
        audit.append(
            f"Activating True System 2 Deliberation: triggers [units={has_units}, "
            f"eq={has_equations}, neg={has_negation}, causal={has_causal}, margin={margin:.3f}]"
        )

        vetoed_indices: List[int] = []
        adjusted_scores = list(probs)

        # Pre-compile formal SCM context when causal triggers fire. When a formal SCM is active,
        # structural equations (e.g. "Y = 2*X") and variable assignments ("Y = 10", "Light = 1")
        # are adjudicated by the SCM rather than standalone SymPy equation solving.
        scm_ctx: Optional[_CausalVerificationContext] = None
        if has_causal:
            scm_ctx = self._build_causal_context(question, q_causal, causal_claims)
            if scm_ctx is not None:
                audit.append(
                    f"SCM constructed with nodes {list(scm_ctx.scm.nodes.keys())}, "
                    f"edges {scm_ctx.edges}: {scm_ctx.query_result.explanation}"
                )

        # Verification 1: Physical Dimensional Analysis (Units)
        if has_units:
            for idx, choice in enumerate(choices):
                subbed_choice = self._substitute_physics_vars(question, choice)
                unit_res = self.unit_verifier.verify_natural_language_claim(question, subbed_choice)
                if not unit_res.is_dimensionally_homogeneous:
                    vetoed_indices.append(idx)
                    adjusted_scores[idx] = -1e9
                    audit.append(f"Choice ({letters[idx]}) VETOED: Incompatible physical dimensions ({unit_res.details})")

        # Verification 2: Mathematical / Algebraic Equivalence & Solving (SymPy)
        q_eq: Optional[str] = None
        alg_target_var: Optional[str] = None
        if has_equations and scm_ctx is None:
            q_eq = self._extract_equation(question)
            tv_match = re.search(
                r"\b(?:solve\s+for|for|find|value\s+of)\s+([a-zA-Z])\b",
                question,
                re.IGNORECASE,
            )
            alg_target_var = tv_match.group(1) if tv_match else None

            for idx, choice in enumerate(choices):
                claim = alg_claims[idx] if idx < len(alg_claims) else None
                verdict, detail = self._verify_algebraic_choice(question, q_eq, alg_target_var, choice, claim)
                if verdict == "confirm":
                    adjusted_scores[idx] += 10.0
                    audit.append(f"Choice ({letters[idx]}) CONFIRMED as {detail}")
                elif verdict == "veto":
                    if idx not in vetoed_indices:
                        vetoed_indices.append(idx)
                    adjusted_scores[idx] = -1e9
                    audit.append(f"Choice ({letters[idx]}) VETOED by AlgebraVerifier: {detail}")

        # Verification 3: Logical Negation Constraints (Z3 & Semantic filter)
        if has_negation:
            for idx, choice in enumerate(choices):
                neg_res = self.logic_verifier.check_negation_consistency(question, choice)
                if not neg_res.is_valid:
                    if idx not in vetoed_indices:
                        vetoed_indices.append(idx)
                    adjusted_scores[idx] = -1e9
                    audit.append(f"Choice ({letters[idx]}) VETOED: {neg_res.details}")

        # Verification 4: Structural Causal Model (Pearl do-calculus & counterfactuals)
        if scm_ctx is not None:
            for idx, choice in enumerate(choices):
                verdict, detail = self._verify_causal_choice(scm_ctx, choice)
                if verdict == "confirm":
                    adjusted_scores[idx] += 10.0
                    audit.append(f"Choice ({letters[idx]}) CONFIRMED by SCM: {detail}")
                elif verdict == "veto":
                    if idx not in vetoed_indices:
                        vetoed_indices.append(idx)
                    adjusted_scores[idx] = -1e9
                    audit.append(f"Choice ({letters[idx]}) VETOED by SCM: {detail}")

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
                vot_expected_utility_gain=vot_decision.expected_utility_gain if vot_decision else None,
                vot_lambda_cost=vot_decision.lambda_cost if vot_decision else None,
                recommended_thinking=vot_decision.recommended_thinking if vot_decision else None,
            )

        # Verification 5: Dynamic Search / Rollout over candidate hypotheses
        # Uses the exact same helpers as Step 2 so MCTS and Step 2 never disagree.
        def combined_verifier(q_text: str, c_text: str) -> Tuple[bool, str]:
            if has_units:
                subbed_c = self._substitute_physics_vars(q_text, c_text)
                res = self.unit_verifier.verify_natural_language_claim(q_text, subbed_c)
                if not res.is_dimensionally_homogeneous:
                    return False, f"Dimensional error: {res.details}"
            if has_equations and scm_ctx is None:
                local_q_eq = q_eq if q_text == question else self._extract_equation(q_text)
                local_tv = alg_target_var
                if q_text != question:
                    m_tv = re.search(
                        r"\b(?:solve\s+for|for|find|value\s+of)\s+([a-zA-Z])\b",
                        q_text,
                        re.IGNORECASE,
                    )
                    local_tv = m_tv.group(1) if m_tv else None
                parsed_c_alg = self.parser.parse_algebraic_claims(q_text, [c_text])
                c_claim = parsed_c_alg[0] if parsed_c_alg else None
                verdict, detail = self._verify_algebraic_choice(q_text, local_q_eq, local_tv, c_text, c_claim)
                if verdict == "veto":
                    return False, detail
            if has_negation:
                n_res = self.logic_verifier.check_negation_consistency(q_text, c_text)
                if not n_res.is_valid:
                    return False, n_res.details
            if scm_ctx is not None:
                verdict, detail = self._verify_causal_choice(scm_ctx, c_text)
                if verdict == "veto":
                    return False, detail
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
            vot_expected_utility_gain=vot_decision.expected_utility_gain if vot_decision else None,
            vot_lambda_cost=vot_decision.lambda_cost if vot_decision else None,
            recommended_thinking=vot_decision.recommended_thinking if vot_decision else None,
        )

    def parse_annotated_prompt(self, raw_text: str) -> Tuple[str, List[str], List[Dict[str, Any]]]:
        """Parse benchmark prompt formatted with QUESTION: and annotated choices [A] [XX tok] Choice.
        
        Strictly protects code blocks (```...```) and inline code structures from false-positive
        matching on assignment statements (e.g. x = 10), returns, or list indexing.
        """
        lines = raw_text.strip().split("\n")
        q_lines = []
        choice_matches = []
        in_question = False
        in_code_block = False

        # Requires explicit choice labeling: [A], (A), A), A., or A: (1-2 chars)
        choice_pattern = re.compile(
            r"^\s*(?:\[([A-Za-z]|\d{1,2})\]|\(([A-Za-z]|\d{1,2})\)|([A-Za-z]|\d{1,2})[).:])\s*(?:\[(\d+)\s*(?:tok|tokens?)\])?\s*(.+)$"
        )

        for line in lines:
            stripped = line.strip()

            # 1. Code block fence tracking - preserve code blocks entirely
            if stripped.startswith("```"):
                in_code_block = not in_code_block
                q_lines.append(line)
                continue
            if in_code_block:
                q_lines.append(line)
                continue

            # 2. Skip horizontal dividers and empty lines
            if not stripped or stripped.startswith("---") or stripped.startswith("==="):
                continue

            # 3. Detect QUESTION: header
            if stripped.upper().startswith("QUESTION:"):
                in_question = True
                q_lines.append(stripped[len("QUESTION:"):].strip())
                continue

            # 4. Check for choice match
            m = choice_pattern.match(stripped)
            if m:
                in_question = False
                choice_matches.append(m)
            elif in_question:
                q_lines.append(line)

        question = "\n".join(q_lines).strip()
        choices = []
        metadata = []
        for m in choice_matches:
            letter = (m.group(1) or m.group(2) or m.group(3)).upper()
            tokens = int(m.group(4)) if m.group(4) else None
            clean_text = m.group(5).strip()
            choices.append(clean_text)
            metadata.append({"letter": letter, "tokens": tokens, "clean_text": clean_text})

        return question, choices, metadata

    def evaluate_annotated_prompt(
        self,
        raw_text: str,
        domain: Optional[str] = None,
        force_sys2: bool = False,
    ) -> DualProcessResult:
        """Evaluate a raw benchmark prompt with QUESTION: and [A] [XX tok] annotations."""
        q, choices, meta = self.parse_annotated_prompt(raw_text)
        if not choices:
            raise ValueError(f"Could not parse any valid candidate choices from raw prompt:\n{raw_text}")
        return self.evaluate(question=q, choices=choices, domain=domain, force_sys2=force_sys2)
