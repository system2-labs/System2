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

    def verify_expression(
        self,
        expression: str,
        context: Optional[str] = None,
    ) -> LogicVerificationResult:
        """Verify a propositional/SMT string expression and optional context via Z3.

        Supports:
        - Natural-language negation checks when context + prose claim are supplied.
        - Entailment via turnstiles (`|=`, `|-`, `⊢`) or a single top-level `=>`/`⇒`.
        - Satisfiability checks over boolean / linear integer constraint lists.
        """
        if not self.available:
            return LogicVerificationResult(
                is_valid=True,
                is_satisfiable=True,
                details="Z3 not available; check bypassed.",
            )

        expr_text = (expression or "").strip()
        if not expr_text:
            return LogicVerificationResult(
                is_valid=False,
                is_satisfiable=False,
                details="Empty logical expression.",
                error="LogicParseError",
            )

        if context is not None and context.strip():
            neg_res = self.check_negation_consistency(context, expr_text)
            if not neg_res.is_valid:
                return neg_res

            # If both context and expression parse as formal Z3 formulas, infer variable
            # sorts jointly across all clauses and verify strict entailment (context |= expr).
            try:
                ctx_parts = _split_top_level(context, separators=(",", ";", "\n"))
                if ctx_parts:
                    ctx_asts = [_parse_logic_ast(p) for p in ctx_parts]
                    expr_ast = _parse_logic_ast(expr_text)
                    ctx_formulas, expr_formulas = _compile_ast_groups([ctx_asts, [expr_ast]])
                    return self.verify_entailment(ctx_formulas, expr_formulas[0])
            except _LogicSortError as exc:
                return LogicVerificationResult(
                    is_valid=False,
                    is_satisfiable=False,
                    details=str(exc),
                    error="LogicTypeError",
                )
            except (ValueError, z3.Z3Exception):
                # Natural language context/claim that does not parse as formal syntax;
                # fall back to the negation consistency verdict above.
                pass

            return neg_res

        # Standalone formal expression: check turnstile entailment first, then top-level '=>',
        # and otherwise treat comma/semicolon-separated clauses as a constraint conjunction.
        turnstile_split = _split_top_level_turnstile(expr_text)
        if turnstile_split is not None:
            premises_raw, conclusion_raw = turnstile_split
            try:
                prem_parts = _split_top_level(premises_raw, separators=(",", ";", "\n")) if premises_raw.strip() else []
                prem_asts = [_parse_logic_ast(p) for p in prem_parts]
                concl_ast = _parse_logic_ast(conclusion_raw)
                premises, conclusions = _compile_ast_groups([prem_asts, [concl_ast]])
                return self.verify_entailment(premises, conclusions[0])
            except _LogicSortError as exc:
                return LogicVerificationResult(
                    is_valid=False,
                    is_satisfiable=False,
                    details=str(exc),
                    error="LogicTypeError",
                )
            except (ValueError, z3.Z3Exception) as exc:
                return LogicVerificationResult(
                    is_valid=False,
                    is_satisfiable=False,
                    details=f"Failed to parse logical entailment: {exc}",
                    error="LogicParseError",
                )

        try:
            parts = _split_top_level(expr_text, separators=(",", ";", "\n"))
            if not parts:
                return LogicVerificationResult(
                    is_valid=False,
                    is_satisfiable=False,
                    details="Empty logical expression.",
                    error="LogicParseError",
                )
            clause_asts = [_parse_logic_ast(p) for p in parts]
            (constraints,) = _compile_ast_groups([clause_asts])
            return self.check_satisfiability(constraints)
        except _LogicSortError as exc:
            return LogicVerificationResult(
                is_valid=False,
                is_satisfiable=False,
                details=str(exc),
                error="LogicTypeError",
            )
        except (ValueError, z3.Z3Exception) as exc:
            return LogicVerificationResult(
                is_valid=False,
                is_satisfiable=False,
                details=f"Failed to parse logical formula: {exc}",
                error="LogicParseError",
            )


# -----------------------------------------------------------------------------
# Safe Recursive-Descent Propositional & Linear SMT Parser (no eval())
# -----------------------------------------------------------------------------

_TOKEN_SPEC = [
    ("SKIP", r"[ \t\r\n]+"),
    ("TURNSTILE", r"\|=|⊢|\|-"),
    ("BICOND", r"<->|<=>|↔|⇔"),
    ("DIMPLIES", r"=>|⇒"),
    ("IMPLIES", r"->|→"),
    ("LE", r"<="),
    ("GE", r">="),
    ("EQ", r"=="),
    ("NE", r"!="),
    ("AND", r"&&|&|∧|/\\"),
    ("OR", r"\|\||\||∨|\\/"),
    ("NOT", r"~|!|¬"),
    ("XOR", r"\^"),
    ("LT", r"<"),
    ("GT", r">"),
    ("PLUS", r"\+"),
    ("MINUS", r"-"),
    ("MUL", r"\*"),
    ("LPAREN", r"\("),
    ("RPAREN", r"\)"),
    ("COMMA", r","),
    ("TRUE", r"⊤"),
    ("FALSE", r"⊥"),
    ("INT", r"\d+"),
    ("IDENT", r"[A-Za-z_][A-Za-z0-9_]*"),
    ("MISMATCH", r"."),
]
_TOKEN_RE = re.compile("|".join(f"(?P<{name}>{pattern})" for name, pattern in _TOKEN_SPEC))
_KEYWORD_KINDS = {
    "and": "AND",
    "or": "OR",
    "not": "NOT",
    "xor": "XOR",
    "iff": "BICOND",
    "true": "TRUE",
    "false": "FALSE",
}


def _tokenize_logic(text: str) -> List[Tuple[str, str]]:
    tokens: List[Tuple[str, str]] = []
    for m in _TOKEN_RE.finditer(text):
        kind = m.lastgroup
        val = m.group()
        if kind == "SKIP":
            continue
        if kind == "MISMATCH":
            raise ValueError(f"Unexpected character '{val}'")
        if kind == "IDENT":
            kw_kind = _KEYWORD_KINDS.get(val.lower())
            if kw_kind is not None:
                kind = kw_kind
        tokens.append((kind, val))
    return tokens


def _split_top_level(text: str, separators: Tuple[str, ...]) -> List[str]:
    """Split text by separators only at parenthesis depth 0."""
    parts: List[str] = []
    depth = 0
    buf: List[str] = []
    for ch in text:
        if ch == "(":
            depth += 1
            buf.append(ch)
        elif ch == ")":
            depth = max(0, depth - 1)
            buf.append(ch)
        elif depth == 0 and ch in separators:
            segment = "".join(buf).strip()
            if segment:
                parts.append(segment)
            buf = []
        else:
            buf.append(ch)
    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts


def _split_top_level_turnstile(text: str) -> Optional[Tuple[str, str]]:
    """Split on a single top-level turnstile (`|=`, `|-`, `⊢`) or unique `=>`/`⇒`."""
    tokens_pos: List[Tuple[int, int, str]] = []
    depth = 0
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                return None
        elif depth == 0:
            two = text[i : i + 2]
            if two in ("|=", "|-"):
                tokens_pos.append((i, i + 2, "TURNSTILE"))
                i += 2
                continue
            if ch == "⊢":
                tokens_pos.append((i, i + 1, "TURNSTILE"))
                i += 1
                continue
            if two == "=>" and (i == 0 or text[i - 1] != "<"):
                tokens_pos.append((i, i + 2, "DIMPLIES"))
                i += 2
                continue
            if ch == "⇒":
                tokens_pos.append((i, i + 1, "DIMPLIES"))
                i += 1
                continue
        i += 1

    turnstiles = [t for t in tokens_pos if t[2] == "TURNSTILE"]
    if len(turnstiles) == 1:
        start, end, _ = turnstiles[0]
        return text[:start].strip(), text[end:].strip()
    if len(turnstiles) > 1:
        return None

    dimplies = [t for t in tokens_pos if t[2] == "DIMPLIES"]
    if len(dimplies) == 1:
        start, end, _ = dimplies[0]
        return text[:start].strip(), text[end:].strip()
    return None


class _LogicSortError(Exception):
    """Raised when variables or equality operands have incompatible Bool/Int sorts."""


@dataclass(frozen=True)
class _LitNode:
    sort: str
    value: Any


@dataclass(frozen=True)
class _VarNode:
    name: str


@dataclass(frozen=True)
class _UnaryNode:
    op: str
    operand: Any


@dataclass(frozen=True)
class _BinNode:
    op: str
    left: Any
    right: Any


@dataclass(frozen=True)
class _CallNode:
    fn: str
    args: List[Any]


class _LogicFormulaParser:
    """Recursive-descent parser converting logic tokens into a typed AST."""

    def __init__(self, tokens: List[Tuple[str, str]]):
        self.tokens = tokens
        self.pos = 0

    def _peek(self) -> Optional[Tuple[str, str]]:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def _consume(self, expected: Optional[str] = None) -> Tuple[str, str]:
        tok = self._peek()
        if tok is None:
            raise ValueError("Unexpected end of expression")
        if expected is not None and tok[0] != expected:
            raise ValueError(f"Expected {expected}, got {tok[0]} ('{tok[1]}')")
        self.pos += 1
        return tok

    def parse(self) -> Any:
        if not self.tokens:
            raise ValueError("Empty expression")
        expr = self._parse_bicond()
        if self._peek() is not None:
            raise ValueError(f"Unexpected token '{self._peek()[1]}'")
        return expr

    def _parse_bicond(self) -> Any:
        left = self._parse_implies()
        while self._peek() and self._peek()[0] == "BICOND":
            self._consume()
            right = self._parse_implies()
            left = _BinNode("BICOND", left, right)
        return left

    def _parse_implies(self) -> Any:
        left = self._parse_or()
        if self._peek() and self._peek()[0] in ("IMPLIES", "DIMPLIES"):
            self._consume()
            right = self._parse_implies()  # right-associative
            return _BinNode("IMPLIES", left, right)
        return left

    def _parse_or(self) -> Any:
        left = self._parse_xor()
        while self._peek() and self._peek()[0] == "OR":
            self._consume()
            right = self._parse_xor()
            left = _BinNode("OR", left, right)
        return left

    def _parse_xor(self) -> Any:
        left = self._parse_and()
        while self._peek() and self._peek()[0] == "XOR":
            self._consume()
            right = self._parse_and()
            left = _BinNode("XOR", left, right)
        return left

    def _parse_and(self) -> Any:
        left = self._parse_not()
        while self._peek() and self._peek()[0] == "AND":
            self._consume()
            right = self._parse_not()
            left = _BinNode("AND", left, right)
        return left

    def _parse_not(self) -> Any:
        if self._peek() and self._peek()[0] == "NOT":
            self._consume()
            return _UnaryNode("NOT", self._parse_not())
        return self._parse_cmp()

    def _parse_cmp(self) -> Any:
        left = self._parse_add()
        tok = self._peek()
        if tok and tok[0] in ("EQ", "NE", "LT", "LE", "GT", "GE"):
            op = self._consume()[0]
            right = self._parse_add()
            return _BinNode(op, left, right)
        return left

    def _parse_add(self) -> Any:
        left = self._parse_mul()
        while self._peek() and self._peek()[0] in ("PLUS", "MINUS"):
            op = self._consume()[0]
            right = self._parse_mul()
            left = _BinNode(op, left, right)
        return left

    def _parse_mul(self) -> Any:
        left = self._parse_unary_arith()
        while self._peek() and self._peek()[0] == "MUL":
            self._consume()
            right = self._parse_unary_arith()
            left = _BinNode("MUL", left, right)
        return left

    def _parse_unary_arith(self) -> Any:
        if self._peek() and self._peek()[0] == "MINUS":
            self._consume()
            return _UnaryNode("NEG", self._parse_unary_arith())
        return self._parse_atom()

    def _parse_atom(self) -> Any:
        tok = self._peek()
        if tok is None:
            raise ValueError("Unexpected end of expression")
        kind, val = tok

        if kind == "LPAREN":
            self._consume("LPAREN")
            inner = self._parse_bicond()
            self._consume("RPAREN")
            return inner
        if kind == "TRUE":
            self._consume()
            return _LitNode("bool", True)
        if kind == "FALSE":
            self._consume()
            return _LitNode("bool", False)
        if kind == "INT":
            self._consume()
            return _LitNode("int", int(val))
        if kind == "IDENT":
            self._consume()
            if self._peek() and self._peek()[0] == "LPAREN":
                return self._parse_call(val)
            return _VarNode(val)

        raise ValueError(f"Unexpected token '{val}'")

    def _parse_call(self, fn_name: str) -> Any:
        self._consume("LPAREN")
        args: List[Any] = []
        if self._peek() and self._peek()[0] != "RPAREN":
            args.append(self._parse_bicond())
            while self._peek() and self._peek()[0] == "COMMA":
                self._consume("COMMA")
                args.append(self._parse_bicond())
        self._consume("RPAREN")

        fname = fn_name.lower()
        if fname in ("and", "or") and len(args) >= 1:
            return _CallNode(fname, args)
        if fname == "not" and len(args) == 1:
            return _CallNode(fname, args)
        if fname in ("implies", "xor", "equiv", "iff") and len(args) == 2:
            return _CallNode(fname, args)
        raise ValueError(f"Unsupported logical function '{fn_name}' with {len(args)} argument(s)")


def _parse_logic_ast(text: str) -> Any:
    return _LogicFormulaParser(_tokenize_logic(text)).parse()


def _collect_vars(node: Any, out: set) -> None:
    if isinstance(node, _VarNode):
        out.add(node.name)
    elif isinstance(node, _UnaryNode):
        _collect_vars(node.operand, out)
    elif isinstance(node, _BinNode):
        _collect_vars(node.left, out)
        _collect_vars(node.right, out)
    elif isinstance(node, _CallNode):
        for arg in node.args:
            _collect_vars(arg, out)


def _resolve_ast_sorts(asts: List[Any]) -> Dict[str, str]:
    """Infer Bool/Int variable sorts across all clauses via fixed-point propagation."""
    var_sorts: Dict[str, str] = {}
    changed = True

    def bind(name: str, sort: str) -> None:
        nonlocal changed
        prev = var_sorts.get(name)
        if prev is not None and prev != sort:
            raise _LogicSortError(f"Variable '{name}' used inconsistently as both {prev} and {sort}.")
        if prev is None:
            var_sorts[name] = sort
            changed = True

    def constrain(node: Any, expected: Optional[str]) -> Optional[str]:
        if isinstance(node, _LitNode):
            if expected is not None and node.sort != expected:
                raise _LogicSortError(f"Expected {expected} expression, got {node.sort} literal '{node.value}'.")
            return node.sort

        if isinstance(node, _VarNode):
            if expected is not None:
                bind(node.name, expected)
            return var_sorts.get(node.name)

        if isinstance(node, _UnaryNode):
            node_sort = "bool" if node.op == "NOT" else "int"
            if expected is not None and node_sort != expected:
                raise _LogicSortError(f"Expected {expected} expression, got {node_sort} operator '{node.op}'.")
            constrain(node.operand, node_sort)
            return node_sort

        if isinstance(node, _CallNode):
            if expected is not None and expected != "bool":
                raise _LogicSortError(f"Expected {expected} expression, got bool function '{node.fn}'.")
            for arg in node.args:
                constrain(arg, "bool")
            return "bool"

        if isinstance(node, _BinNode):
            if node.op in ("BICOND", "IMPLIES", "OR", "XOR", "AND"):
                if expected is not None and expected != "bool":
                    raise _LogicSortError(f"Expected {expected} expression, got bool operator '{node.op}'.")
                constrain(node.left, "bool")
                constrain(node.right, "bool")
                return "bool"

            if node.op in ("PLUS", "MINUS", "MUL"):
                if expected is not None and expected != "int":
                    raise _LogicSortError(f"Expected {expected} expression, got int operator '{node.op}'.")
                constrain(node.left, "int")
                constrain(node.right, "int")
                return "int"

            if node.op in ("LT", "LE", "GT", "GE"):
                if expected is not None and expected != "bool":
                    raise _LogicSortError(f"Expected {expected} expression, got bool comparison '{node.op}'.")
                constrain(node.left, "int")
                constrain(node.right, "int")
                return "bool"

            if node.op in ("EQ", "NE"):
                if expected is not None and expected != "bool":
                    raise _LogicSortError(f"Expected {expected} expression, got bool comparison '{node.op}'.")
                ls = constrain(node.left, None)
                rs = constrain(node.right, None)
                if ls is not None and rs is not None and ls != rs:
                    raise _LogicSortError(f"Incompatible operand sorts for '{node.op}': {ls} vs {rs}.")
                if ls is not None and rs is None:
                    constrain(node.right, ls)
                elif rs is not None and ls is None:
                    constrain(node.left, rs)
                return "bool"

        raise ValueError(f"Unsupported AST node: {node!r}")

    # Propagate sort constraints across all clauses until fixed point so cross-clause
    # equalities (e.g. "x == y, x > 5") resolve every linked variable to Int.
    while changed:
        changed = False
        for ast in asts:
            constrain(ast, "bool")

    # Variables untouched by integer/arithmetic constraints default to Bool.
    all_vars: set = set()
    for ast in asts:
        _collect_vars(ast, all_vars)
    for name in all_vars:
        var_sorts.setdefault(name, "bool")

    # Re-verify every AST with concrete sorts to catch any residual Bool/Int mismatch.
    for ast in asts:
        constrain(ast, "bool")

    return var_sorts


def _lower_ast(node: Any, var_sorts: Dict[str, str], symtab: Dict[str, Any]) -> Any:
    if isinstance(node, _LitNode):
        return z3.BoolVal(node.value) if node.sort == "bool" else z3.IntVal(node.value)

    if isinstance(node, _VarNode):
        sym = symtab.get(node.name)
        if sym is None:
            sym = z3.Int(node.name) if var_sorts[node.name] == "int" else z3.Bool(node.name)
            symtab[node.name] = sym
        return sym

    if isinstance(node, _UnaryNode):
        arg = _lower_ast(node.operand, var_sorts, symtab)
        return z3.Not(arg) if node.op == "NOT" else -arg

    if isinstance(node, _CallNode):
        args = [_lower_ast(a, var_sorts, symtab) for a in node.args]
        if node.fn == "and":
            return z3.And(*args) if len(args) > 1 else args[0]
        if node.fn == "or":
            return z3.Or(*args) if len(args) > 1 else args[0]
        if node.fn == "not":
            return z3.Not(args[0])
        if node.fn == "implies":
            return z3.Implies(args[0], args[1])
        if node.fn == "xor":
            return z3.Xor(args[0], args[1])
        return args[0] == args[1]

    if isinstance(node, _BinNode):
        lval = _lower_ast(node.left, var_sorts, symtab)
        rval = _lower_ast(node.right, var_sorts, symtab)
        op = node.op
        if op == "BICOND" or op == "EQ":
            return lval == rval
        if op == "NE":
            return lval != rval
        if op == "IMPLIES":
            return z3.Implies(lval, rval)
        if op == "OR":
            return z3.Or(lval, rval)
        if op == "XOR":
            return z3.Xor(lval, rval)
        if op == "AND":
            return z3.And(lval, rval)
        if op == "LT":
            return lval < rval
        if op == "LE":
            return lval <= rval
        if op == "GT":
            return lval > rval
        if op == "GE":
            return lval >= rval
        if op == "PLUS":
            return lval + rval
        if op == "MINUS":
            return lval - rval
        if op == "MUL":
            return lval * rval

    raise ValueError(f"Unsupported AST node: {node!r}")


def _compile_ast_groups(ast_groups: List[List[Any]]) -> List[List[Any]]:
    flat = [ast for group in ast_groups for ast in group]
    var_sorts = _resolve_ast_sorts(flat)
    symtab: Dict[str, Any] = {}
    return [[_lower_ast(ast, var_sorts, symtab) for ast in group] for group in ast_groups]
