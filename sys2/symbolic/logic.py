"""logic.py - Exact Constraint Satisfaction & Logical Entailment Engine.

Enforces formal propositional & first-order logic constraints using Z3 SMT solver.
Solves constraint satisfaction problems, checks whether premises strictly entail
a conclusion (via UNSAT check on Premises ∧ ¬Conclusion), and catches counterexamples
or direct negation contradictions that feedforward embeddings miss.
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

try:
    import z3
    _Z3_AVAILABLE = True
except ImportError:
    _Z3_AVAILABLE = False


@dataclass
class LogicVerificationResult:
    """Outcome of logical constraint checking or entailment verification."""
    is_valid: bool
    is_satisfiable: bool
    counterexample: Optional[Dict[str, str]] = None
    details: str = ""
    error: Optional[str] = None


class LogicVerifier:
    """Verifies logical entailment and constraint satisfaction via Z3."""

    def __init__(self, timeout_ms: int = 2000):
        self.available = _Z3_AVAILABLE
        self.timeout_ms = timeout_ms

    def check_satisfiability(self, constraints: List[Any]) -> LogicVerificationResult:
        """Check if a list of Z3 boolean constraints is simultaneously satisfiable."""
        if not self.available:
            return LogicVerificationResult(
                is_valid=True,
                is_satisfiable=True,
                details="Z3 not available; check bypassed.",
            )

        solver = z3.Solver()
        solver.set("timeout", self.timeout_ms)
        for c in constraints:
            solver.add(c)

        check_res = solver.check()
        if check_res == z3.sat:
            m = solver.model()
            model_dict = {str(d): str(m[d]) for d in m.decls()}
            return LogicVerificationResult(
                is_valid=True,
                is_satisfiable=True,
                counterexample=model_dict,
                details=f"Satisfiable. Model: {model_dict}",
            )
        elif check_res == z3.unsat:
            return LogicVerificationResult(
                is_valid=False,
                is_satisfiable=False,
                details="Unsatisfiable (contradiction found among constraints).",
            )
        else:
            return LogicVerificationResult(
                is_valid=False,
                is_satisfiable=False,
                details=f"Z3 returned unknown/timeout after {self.timeout_ms}ms.",
                error="TimeoutOrUnknown",
            )

    def verify_entailment(
        self,
        premises: List[Any],
        conclusion: Any,
    ) -> LogicVerificationResult:
        """Check if premises strictly entail conclusion.

        In classical logic: Premises |= Conclusion iff (Premises AND NOT Conclusion) is UNSAT.
        Requires premises themselves to be satisfiable (preventing vacuous truth via ex falso quodlibet).
        """
        if not self.available:
            return LogicVerificationResult(
                is_valid=True,
                is_satisfiable=True,
                details="Z3 not available; check bypassed.",
            )

        # 1. Verify that premises themselves are consistent
        prem_solver = z3.Solver()
        prem_solver.set("timeout", self.timeout_ms)
        for p in premises:
            prem_solver.add(p)
        prem_check = prem_solver.check()
        if prem_check == z3.unsat:
            return LogicVerificationResult(
                is_valid=False,
                is_satisfiable=False,
                details="Premises are mutually contradictory (ex falso quodlibet). Entailment is vacuous.",
                error="ContradictoryPremises",
            )
        elif prem_check != z3.sat:
            return LogicVerificationResult(
                is_valid=False,
                is_satisfiable=False,
                details="Z3 timed out evaluating premise consistency.",
                error="Timeout",
            )

        # 2. Check Premises AND NOT Conclusion
        solver = z3.Solver()
        solver.set("timeout", self.timeout_ms)
        for p in premises:
            solver.add(p)
        solver.add(z3.Not(conclusion))

        check_res = solver.check()
        if check_res == z3.unsat:
            # No counterexample exists -> entailment holds!
            return LogicVerificationResult(
                is_valid=True,
                is_satisfiable=True,
                details="Strict logical entailment verified (¬Conclusion is UNSAT).",
            )
        elif check_res == z3.sat:
            # A counterexample was found -> entailment fails, but formula is satisfiable!
            m = solver.model()
            counterexample = {str(d): str(m[d]) for d in m.decls()}
            return LogicVerificationResult(
                is_valid=False,
                is_satisfiable=True,
                counterexample=counterexample,
                details=f"Entailment FAILS. Counterexample found: {counterexample}",
            )
        else:
            return LogicVerificationResult(
                is_valid=False,
                is_satisfiable=False,
                details=f"Z3 returned unknown/timeout after {self.timeout_ms}ms.",
                error="TimeoutOrUnknown",
            )

    # Negation markers ordered longest-first so multi-word forms match before bare "not"/"no".
    _DECL_NEG_RE = re.compile(
        r"\b(?P<marker>"
        r"does\s+not|do\s+not|did\s+not|doesn't|don't|didn't|"
        r"is\s+not|are\s+not|was\s+not|were\s+not|isn't|aren't|wasn't|weren't|"
        r"cannot|can\s+not|can't|could\s+not|couldn't|"
        r"must\s+not|mustn't|should\s+not|shouldn't|"
        r"will\s+not|won't|would\s+not|wouldn't|"
        r"may\s+not|might\s+not|shall\s+not|"
        r"has\s+not|have\s+not|had\s+not|hasn't|haven't|hadn't|"
        r"no\s+longer|unable\s+to|fails?\s+to|failed\s+to|never|neither|not|no"
        r")\s+"
        r"(?:(?:be|been|being|ever|directly|safely|properly|freely|automatically)\s+)?"
        r"(?P<w1>[a-z]{4,})"
        r"(?:\s+(?:(?:the|a|an|any|all|its|their|our|this|that)\s+)?(?P<w2>[a-z]{4,}))?",
        re.IGNORECASE,
    )

    # Cues appearing before a target term in the same clause that negate, prevent, or contrast it.
    _CLAIM_NEG_PRE_RE = re.compile(
        r"\b(?:"
        r"not|no|never|cannot|can\s+not|can't|don't|doesn't|didn't|"
        r"isn't|aren't|wasn't|weren't|won't|wouldn't|shouldn't|mustn't|couldn't|"
        r"haven't|hasn't|hadn't|neither|nor|none|nothing|no\s+longer|"
        r"prevents?|prevented|preventing|stops?|stopped|stopping|"
        r"blocks?|blocked|blocking|avoids?|avoided|avoiding|"
        r"forbids?|forbidden|prohibits?|prohibited|"
        r"refuses?|refused|rejects?|rejected|denies?|denied|"
        r"lacks?|lacked|lacking|lack\s+of|"
        r"fails?\s+to|failed\s+to|unable\s+to|incapable\s+of|unlikely\s+to|"
        r"instead\s+of|rather\s+than|as\s+opposed\s+to|without|except|excluding|"
        r"unlike|contrary\s+to|far\s+from|free\s+of|free\s+from|"
        r"inhibits?|inhibited|resists?|resisted|"
        r"allergic\s+to|contraindicated|intolerant\s+of|"
        r"keeps?\s+\w+\s+from|kept\s+\w+\s+from|protects?\s+\w+\s+from|"
        r"why|whether|even\s+if"
        r")\b",
        re.IGNORECASE,
    )

    # Passive or post-predicate negation cues appearing after the target term in the same clause.
    _CLAIM_NEG_POST_RE = re.compile(
        r"\b(?:"
        r"(?:is|are|was|were|remains?|stays?)\s+(?:strictly\s+|completely\s+|entirely\s+)?"
        r"(?:not|never|prevented|avoided|prohibited|forbidden|blocked|impossible|"
        r"absent|missing|depleted|unavailable|contraindicated|disallowed|eliminated)|"
        r"(?:does|do|did|will|would|can|could|must|should)\s+not|"
        r"doesn't|don't|didn't|won't|wouldn't|can't|cannot|never|fails?\s+to"
        r")\b",
        re.IGNORECASE,
    )

    # Light/support verbs whose negated meaning lives on their direct object (e.g. "cannot take penicillin").
    _LIGHT_VERBS = {
        "take", "use", "have", "get", "make", "give", "allow", "permit",
        "accept", "receive", "consume", "ingest", "include", "contain",
    }

    # Hedges/quantifiers after "not" that do not express an absolute prohibition (e.g. "not always X").
    _HEDGE_WORDS = {
        "always", "necessarily", "only", "just", "merely", "completely",
        "entirely", "fully", "quite", "very", "enough", "really", "actually",
        "every", "each", "most", "many", "much", "some", "more", "less",
    }

    # Function words and prepositions that must not be treated as negated predicates or direct objects.
    _STOP_WORDS = {
        "that", "this", "these", "those", "with", "from", "have", "were", "been",
        "will", "would", "could", "should", "some", "more", "less", "much", "many",
        "notice", "time", "make", "take", "give", "part", "case", "good", "year",
        "work", "state", "under", "over", "into", "onto", "during", "before",
        "after", "while", "when", "where", "than", "then", "because", "since",
        "until", "unless", "through", "across", "between", "among", "about",
        "around", "above", "below", "near", "here", "there", "also", "still",
        "already", "even", "ever", "both", "each", "same", "able", "used",
        "does", "done", "true", "false", "well", "such", "without", "within",
        "only", "just", "very", "being", "their", "them", "they", "what", "which",
    }

    @staticmethod
    def _is_interrogative_sentence(sentence: str) -> bool:
        """Return True for questions and negative fill-in-the-blank stems.

        Interrogative prompts like 'Why does ice not sink?' or 'Which gas is NOT produced?'
        ask the reader to explain or select the negated property rather than stating a
        declarative prohibition that forbids mentioning the term in the answer.
        """
        s = sentence.strip()
        if not s:
            return True
        if s.endswith("?"):
            return True
        # Negative fill-in-the-blank stems (e.g., "... CANNOT be:" or "... CANNOT:") ask
        # the candidate choice to name the prohibited/excluded property itself.
        if re.search(
            r"\b(?:cannot|can\s+not|must\s+not|should\s+not|will\s+not|"
            r"does\s+not|do\s+not|is\s+not|are\s+not|never)\s*(?:be|have|do)?\s*:\s*$",
            s,
            re.IGNORECASE,
        ):
            return True
        # Direct or comma-prefixed wh-questions / selection prompts (even without trailing '?').
        if re.match(
            r"^(?:why|which|what|who|whom|whose|where|when|how|"
            r"select|choose|identify|name|determine)\b",
            s,
            re.IGNORECASE,
        ):
            return True
        if re.search(r",\s*(?:why|which|what|who|whom|whose|where|how)\b", s, re.IGNORECASE):
            return True
        # Polar auxiliary-first questions lacking '?': "Does ice not sink..."
        if re.match(
            r"^(?:does|do|did|is|are|was|were|can|could|would|should|will)\s+[a-z]",
            s,
            re.IGNORECASE,
        ):
            return True
        return False

    @staticmethod
    def _word_inflections(word: str) -> List[str]:
        """Generate strict morphological forms for a predicate/target word."""
        forms = [word]
        if word.endswith(("ed", "ing")):
            return forms
        forms.append(f"{word}s")
        forms.append(f"{word}es")
        if word.endswith("e") and len(word) >= 5:
            forms.append(f"{word}d")
            forms.append(f"{word[:-1]}ing")
        else:
            forms.append(f"{word}ed")
            forms.append(f"{word}ing")
        return forms

    def check_negation_consistency(self, context_text: str, candidate_claim: str) -> LogicVerificationResult:
        """Audit candidate claims for direct contradictions against declarative context negations.

        Extracts prohibitions/negations ONLY from declarative context sentences (skipping
        interrogative prompts and hypothetical 'if'-clauses), and checks whether the
        candidate claim affirmatively asserts the negated predicate/target without local
        negation, contrast, or prevention framing (e.g. 'prevents', 'rather than', 'avoids').
        """
        if not context_text or not candidate_claim:
            return LogicVerificationResult(
                is_valid=True,
                is_satisfiable=True,
                details="No overt negation conflict detected.",
            )

        # 1. Extract declarative negation constraints from non-interrogative context sentences.
        sentences = [s.strip() for s in re.findall(r"[^.!?\n]+[.!?]?", context_text) if s.strip()]
        constraints: List[Tuple[str, str, Optional[str], str]] = []

        for sent in sentences:
            if self._is_interrogative_sentence(sent):
                continue

            # Strip leading hypothetical condition ("If X does not ..., Y happens") so
            # negations inside the antecedent do not become unconditional prohibitions.
            decl_part = sent
            if re.match(r"^(?:if|suppose|assuming|unless|whenever|provided\s+that)\b", decl_part, re.IGNORECASE):
                comma_idx = decl_part.find(",")
                if comma_idx == -1:
                    continue
                decl_part = decl_part[comma_idx + 1 :].strip()

            lower_decl = decl_part.lower()
            for m in self._DECL_NEG_RE.finditer(lower_decl):
                marker = m.group("marker").strip()
                w1 = m.group("w1").lower()
                w2 = m.group("w2").lower() if m.group("w2") else None

                if w1 in self._HEDGE_WORDS:
                    continue

                if w1 in self._LIGHT_VERBS:
                    # For light verbs ("cannot take penicillin"), the constrained item is the object.
                    if w2 and len(w2) >= 5 and w2 not in self._STOP_WORDS:
                        constraints.append((marker, w2, None, f"{marker} {w1} {w2}"))
                    continue

                if w1 in self._STOP_WORDS:
                    continue

                obj = w2 if (w2 and len(w2) >= 4 and w2 not in self._STOP_WORDS) else None
                phrase = f"{marker} {w1}" + (f" {obj}" if obj else "")
                constraints.append((marker, w1, obj, phrase))

        if not constraints:
            return LogicVerificationResult(
                is_valid=True,
                is_satisfiable=True,
                details="No declarative negation constraints detected.",
            )

        # 2. Check whether candidate_claim affirmatively asserts any negated target.
        lower_claim = candidate_claim.lower()
        clause_bounds = [0] + [m.end() for m in re.finditer(r"[.!?;,\n]+", lower_claim)] + [len(lower_claim)]

        for _marker, target, obj, phrase in constraints:
            forms = self._word_inflections(target)
            target_re = re.compile(r"\b(?:" + "|".join(re.escape(f) for f in forms) + r")\b")

            for match in target_re.finditer(lower_claim):
                # Locate the enclosing clause for this occurrence.
                c_start = 0
                c_end = len(lower_claim)
                for b in clause_bounds:
                    if b <= match.start():
                        c_start = b
                    if b >= match.end():
                        c_end = b
                        break

                pre_clause = lower_claim[c_start : match.start()]
                post_clause = lower_claim[match.end() : c_end]

                # Trim pre_clause at contrast conjunctions or fresh affirmative conjuncts
                # ("is not open and is closed") so an earlier negation does not mask a new clause.
                split_matches = list(
                    re.finditer(
                        r"\b(?:but|however|whereas|while|although|yet|"
                        r"and\s+(?:is|are|was|were|will|should|must|can|would|does|do))\b",
                        pre_clause,
                    )
                )
                if split_matches:
                    pre_clause = pre_clause[split_matches[-1].end() :]

                # Local negation, prevention, avoidance, or contrast framing makes the mention consistent.
                if self._CLAIM_NEG_PRE_RE.search(pre_clause[-80:]):
                    continue
                if self._CLAIM_NEG_POST_RE.search(post_clause[:60]):
                    continue

                # If the constraint restricted a verb to a specific object (e.g. "not conduct electricity",
                # "never delete production"), verify the claim targets that object (or a bare/pronoun
                # reference) rather than an unrelated object (e.g. "conduct heat", "delete cache").
                if obj is not None:
                    clause_text = lower_claim[c_start:c_end]
                    obj_present = bool(re.search(rf"\b{re.escape(obj)}s?\b", clause_text))
                    if not obj_present:
                        post_tokens = re.findall(r"\b[a-z]+\b", post_clause)
                        generic_after = (
                            not post_tokens
                            or post_tokens[0]
                            in {
                                "it", "them", "this", "that", "all", "everything", "anything",
                                "in", "at", "on", "under", "during", "well", "easily",
                                "freely", "properly", "immediately", "automatically", "again",
                            }
                        )
                        if not generic_after:
                            continue

                return LogicVerificationResult(
                    is_valid=False,
                    is_satisfiable=False,
                    details=(
                        f"Direct negation conflict detected: context specifies '{phrase}', "
                        f"but choice affirmatively asserts '{match.group(0)}'."
                    ),
                )

        return LogicVerificationResult(
            is_valid=True,
            is_satisfiable=True,
            details="No overt negation conflict detected.",
        )
