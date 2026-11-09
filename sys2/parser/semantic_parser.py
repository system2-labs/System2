"""semantic_parser.py - Domain-Structured Semantic Parser & Formal DSL Slot Extractor.

Translates unstructured natural language questions and multiple-choice hypotheses into
typed, domain-structured Abstract Syntax Trees (PhysicalClaim, AlgebraicClaim,
LogicalClaim, CausalClaim) for formal verification via Pint, SymPy, Z3, and SCM.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

try:
    import pint
    _UREF = pint.UnitRegistry()
    _PINT_AVAILABLE = True
except ImportError:
    _UREF = None
    _PINT_AVAILABLE = False

from sys2.parser.dsl import (
    PhysicalClaim,
    AlgebraicClaim,
    LogicalClaim,
    CausalClaim,
)


class SemanticParser:
    """Extracts formal verification claims from natural language questions and options."""

    TARGET_CONCEPTS: Dict[str, str] = {
        "kinetic energy": "joule",
        "potential energy": "joule",
        "work": "joule",
        "energy": "joule",
        "force": "newton",
        "power": "watt",
        "acceleration": "meter / second^2",
        "velocity": "meter / second",
        "speed": "meter / second",
        "mass": "kilogram",
        "resistance": "ohm",
        "pressure": "pascal",
        "voltage": "volt",
        "current": "ampere",
        "distance": "meter",
        "length": "meter",
        "time": "second",
        "momentum": "kilogram * meter / second",
        "frequency": "hertz",
        "density": "kilogram / meter^3",
    }

    COMPOUND_UNIT_MAPPINGS: List[Tuple[re.Pattern, str]] = [
        # Force: kilogram meter per second squared -> N
        (re.compile(r"\bkilogram\s+meters?\s+per\s+second\s+squared\b", re.IGNORECASE), "N"),
        (re.compile(r"\bkilogram\s+metres?\s+per\s+second\s+squared\b", re.IGNORECASE), "N"),
        (re.compile(r"\bkg\s*\*\s*m\s*/\s*s\^2\b", re.IGNORECASE), "N"),
        (re.compile(r"\bkg\s*\*\s*m\s*/\s*s²\b", re.IGNORECASE), "N"),
        # Energy / Work: kilogram meter squared per second squared -> joule
        (re.compile(r"\bkilogram\s+meters?\s+squared\s+per\s+second\s+squared\b", re.IGNORECASE), "joule"),
        (re.compile(r"\bkilogram\s+metres?\s+squared\s+per\s+second\s+squared\b", re.IGNORECASE), "joule"),
        (re.compile(r"\bnewton\s+meters?\b", re.IGNORECASE), "joule"),
        (re.compile(r"\bnewton\s+metres?\b", re.IGNORECASE), "joule"),
        (re.compile(r"\bN\s*[*·]\s*m\b", re.IGNORECASE), "joule"),
        # Acceleration: meter per second squared -> meter / second^2 (MUST be before velocity!)
        (re.compile(r"\bmeters?\s+per\s+second\s+squared\b", re.IGNORECASE), "meter / second^2"),
        (re.compile(r"\bmetres?\s+per\s+second\s+squared\b", re.IGNORECASE), "meter / second^2"),
        # Velocity / Speed: meter per second -> m/s
        (re.compile(r"\bmeters?\s+per\s+second\b", re.IGNORECASE), "m/s"),
        (re.compile(r"\bmetres?\s+per\s+second\b", re.IGNORECASE), "m/s"),
        (re.compile(r"\bkilometers?\s+per\s+hour\b", re.IGNORECASE), "km/h"),
        (re.compile(r"\bkilometres?\s+per\s+hour\b", re.IGNORECASE), "km/h"),
        (re.compile(r"\bmiles?\s+per\s+hour\b", re.IGNORECASE), "mph"),
        # Power: joule per second -> watt
        (re.compile(r"\bjoules?\s+per\s+second\b", re.IGNORECASE), "watt"),
        # Pressure: newton per square meter -> pascal
        (re.compile(r"\bnewtons?\s+per\s+square\s+meters?\b", re.IGNORECASE), "pascal"),
        (re.compile(r"\bnewtons?\s+per\s+square\s+metres?\b", re.IGNORECASE), "pascal"),
        (re.compile(r"\bnewtons?\s+per\s+meter\s+squared\b", re.IGNORECASE), "pascal"),
        # Momentum
        (re.compile(r"\bkilogram\s+meters?\s+per\s+second\b", re.IGNORECASE), "kilogram * meter / second"),
        (re.compile(r"\bkilogram\s+metres?\s+per\s+second\b", re.IGNORECASE), "kilogram * meter / second"),
        # Density
        (re.compile(r"\bkilograms?\s+per\s+cubic\s+meters?\b", re.IGNORECASE), "kilogram / meter^3"),
        (re.compile(r"\bkilograms?\s+per\s+cubic\s+metres?\b", re.IGNORECASE), "kilogram / meter^3"),
        # Electrical
        (re.compile(r"\bcoulombs?\s+per\s+second\b", re.IGNORECASE), "ampere"),
        (re.compile(r"\bvolts?\s+per\s+amperes?\b", re.IGNORECASE), "ohm"),
        (re.compile(r"\bvolts?\s+per\s+meters?\b", re.IGNORECASE), "volt / meter"),
    ]

    NEGATION_MARKERS = [
        "not", "never", "cannot", "neither", "no", "unable to", "fails to", "except", "without"
    ]

    STOP_WORDS = {
        "that", "this", "with", "from", "have", "were", "been", "will", "would",
        "could", "should", "some", "more", "less", "much", "many", "notice", "time",
        "make", "take", "give", "part", "case", "good", "year", "work", "state",
        "the", "a", "an", "is", "are", "be", "to", "in", "of", "and", "or",
    }

    def __init__(self):
        self.ureg = _UREF if _PINT_AVAILABLE else None

    def normalize_units(self, text: str) -> str:
        """Replace verbose natural language compound unit names with canonical symbols."""
        if not text:
            return ""
        normalized = text
        for pattern, replacement in self.COMPOUND_UNIT_MAPPINGS:
            normalized = pattern.sub(replacement, normalized)
        return normalized

    def resolve_target_concept(self, text: str) -> Optional[Tuple[str, str]]:
        """Identify physical concept and canonical unit from text.

        Returns (concept_name, canonical_unit) or None.
        Matches longer phrases (e.g. 'kinetic energy') before single words (e.g. 'energy').
        """
        if not text:
            return None
        lower = text.lower()
        for concept in sorted(self.TARGET_CONCEPTS.keys(), key=len, reverse=True):
            if re.search(rf"\b{re.escape(concept)}\b", lower):
                return concept, self.TARGET_CONCEPTS[concept]
        return None

    def parse_quantity(self, expr_str: str) -> Optional[Any]:
        """Safely parse a physical quantity string into a Pint Quantity."""
        if not self.ureg or not expr_str:
            return None
        cleaned = self.normalize_units(expr_str.strip().strip("$").strip())
        cleaned = cleaned.replace("×", "*").replace("·", "*").replace("²", "^2")

        # Check pure concept name
        lower = cleaned.lower()
        if lower in self.TARGET_CONCEPTS:
            unit_name = self.TARGET_CONCEPTS[lower]
            try:
                return self.ureg(f"1 * {unit_name}")
            except Exception:
                pass

        # Protect division like '/ 5 m' -> '/ (5 * m)'
        cleaned = re.sub(r"/\s*(\d+)\s*([a-zA-Z]+)", r"/ (\1 * \2)", cleaned)

        try:
            return self.ureg(cleaned)
        except Exception:
            return None

    def _extract_quantities_from_text(self, text: str) -> List[Any]:
        """Extract physical quantities from text as Pint Quantities or raw strings."""
        if not text:
            return []
        norm_text = self.normalize_units(text)
        quantities: List[Any] = []

        # Find number + unit tokens
        matches = list(re.finditer(r"\b(\d+(?:\.\d+)?)\s*([a-zA-ZΩµ/^²·*]+)\b", norm_text))
        if matches:
            for m in matches:
                q_str = m.group(0).strip()
                parsed_q = self.parse_quantity(q_str)
                if parsed_q is not None and hasattr(parsed_q, "dimensionality"):
                    quantities.append(parsed_q)
                else:
                    quantities.append(q_str)
        else:
            # Check if text contains a standalone quantity or concept
            parsed_q = self.parse_quantity(norm_text)
            if parsed_q is not None and hasattr(parsed_q, "dimensionality"):
                quantities.append(parsed_q)

        return quantities

    def parse_physical_claims(
        self,
        question: str,
        choices: List[str],
    ) -> Tuple[Optional[str], List[PhysicalClaim]]:
        """Extract formal PhysicalClaim representations from question and candidate choices.

        Returns (target_concept_from_question, list_of_claims_per_choice).
        """
        q_concept_info = self.resolve_target_concept(question)
        target_concept = q_concept_info[0] if q_concept_info else None
        target_dim = q_concept_info[1] if q_concept_info else None

        claims: List[PhysicalClaim] = []

        for choice in choices:
            norm_choice = self.normalize_units(choice)
            claimed_relation: Optional[str] = None
            choice_concept: Optional[str] = None
            choice_dim: Optional[str] = None

            # Detect explicit equality or relation
            if "=" in norm_choice or "==" in norm_choice:
                claimed_relation = "==" if "==" in norm_choice else "="
                parts = re.split(r"==|=", norm_choice, maxsplit=1)
                lhs, rhs = parts[0].strip(), parts[1].strip()

                # Check if LHS is a target concept
                lhs_concept_info = self.resolve_target_concept(lhs)
                if lhs_concept_info:
                    choice_concept = lhs_concept_info[0]
                    choice_dim = lhs_concept_info[1]

                quantities = self._extract_quantities_from_text(rhs)
                if not quantities:
                    quantities = self._extract_quantities_from_text(norm_choice)
            else:
                # Check relation keywords like 'is', 'equals'
                is_match = re.search(r"\b(is|equals)\b", norm_choice, re.IGNORECASE)
                if is_match:
                    claimed_relation = is_match.group(1).lower()

                # Check if choice mentions a specific concept
                c_info = self.resolve_target_concept(norm_choice)
                if c_info:
                    choice_concept = c_info[0]
                    choice_dim = c_info[1]

                quantities = self._extract_quantities_from_text(norm_choice)

            # Inherit from question if not specified in choice
            final_concept = choice_concept or target_concept
            final_dim = choice_dim or target_dim

            claims.append(
                PhysicalClaim(
                    target_concept=final_concept,
                    target_dimension=final_dim,
                    extracted_quantities=quantities,
                    claimed_relation=claimed_relation,
                )
            )

        return target_concept, claims

    def parse_algebraic_claims(
        self,
        question: str,
        choices: List[str],
    ) -> List[Optional[AlgebraicClaim]]:
        """Extract AlgebraicClaim ASTs from choices and optional equation in question."""
        # Detect target variable in question (e.g., 'solve for x', 'value of y')
        q_var_match = re.search(
            r"\b(?:solve\s+for|value\s+of|find)\s+([a-zA-Z])\b",
            question,
            re.IGNORECASE,
        )
        target_var = q_var_match.group(1) if q_var_match else None

        # Detect equation in question
        q_has_eq = bool("=" in question or "==" in question)

        claims: List[Optional[AlgebraicClaim]] = []

        for choice in choices:
            c_str = choice.strip()
            # Explicit equality in choice: '(x + 1)^2 == x^2 + 2*x + 1' or 'y = 2*x + 1'
            if "=" in c_str or "==" in c_str:
                relation = "==" if "==" in c_str else "="
                parts = re.split(r"==|=", c_str, maxsplit=1)
                lhs = parts[0].strip()
                rhs = parts[1].strip()

                var = target_var
                if not var and re.match(r"^[a-zA-Z]$", lhs):
                    var = lhs

                claims.append(
                    AlgebraicClaim(
                        lhs=lhs,
                        rhs=rhs,
                        relation=relation,
                        target_var=var,
                    )
                )
            elif target_var and q_has_eq:
                # Choice gives the value or expression for the target variable: e.g. '3'
                claims.append(
                    AlgebraicClaim(
                        lhs=target_var,
                        rhs=c_str,
                        relation="==",
                        target_var=target_var,
                    )
                )
            else:
                claims.append(None)

        return claims

    def parse_logical_claims(
        self,
        question: str,
        choices: List[str],
    ) -> List[LogicalClaim]:
        """Extract LogicalClaim ASTs with premises, conclusion, and negation slots."""
        # Extract premises from question text
        raw_sentences = re.split(r"[.;\n]+", question)
        premises = []
        for s in raw_sentences:
            s_clean = s.strip()
            if not s_clean:
                continue
            # Omit final question prompt if it's a generic inquiry
            if s_clean.endswith("?") or re.match(r"^(?:which|what|select|choose|find)\b", s_clean, re.I):
                continue
            premises.append(s_clean)

        if not premises and question.strip():
            premises = [question.strip()]

        lower_q = question.lower()
        claims: List[LogicalClaim] = []

        for choice in choices:
            lower_c = choice.lower()
            combined_text = f"{lower_q} {lower_c}"

            # Check if negation is active
            has_neg = any(
                re.search(rf"\b{re.escape(m)}\b", combined_text)
                for m in self.NEGATION_MARKERS
            )

            # Extract negated terms
            negated_terms: List[str] = []
            for marker in ["not", "never", "cannot", "neither", "no", "unable to", "fails to"]:
                pattern = rf"\b{re.escape(marker)}\s+([a-zA-Z0-9_-]+(?:\s+[a-zA-Z0-9_-]+)?)"
                for m in re.finditer(pattern, combined_text):
                    term = m.group(1).strip()
                    first_word = term.split()[0]
                    if first_word not in self.STOP_WORDS and len(first_word) >= 3:
                        if first_word not in negated_terms:
                            negated_terms.append(first_word)
                        if term not in negated_terms and term != first_word:
                            negated_terms.append(term)

            claims.append(
                LogicalClaim(
                    premises=list(premises),
                    conclusion=choice.strip(),
                    has_negation=has_neg,
                    negated_terms=negated_terms,
                )
            )

        return claims

    def parse_causal_claims(
        self,
        question: str,
        choices: List[str],
    ) -> Tuple[Optional[CausalClaim], List[CausalClaim]]:
        """Extract CausalClaim ASTs (interventions, counterfactuals, cause-effect graph).

        Returns (question_causal_claim_or_None, list_of_claims_per_choice).
        """
        def _extract_causal_slots(text: str) -> Tuple[List[str], List[str], Dict[str, Any], bool]:
            lower = text.lower()
            is_cf = any(
                w in lower
                for w in ["counterfactual", "would have been", "had been", "what if", "had not been", "would be if"]
            )

            # Pearl's do-calculus interventions: do(Var = Val)
            interventions: Dict[str, Any] = {}
            for m in re.finditer(r"do\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^)]+)\)", text, re.IGNORECASE):
                var = m.group(1).strip()
                val_str = m.group(2).strip()
                try:
                    val: Any = int(val_str)
                except ValueError:
                    try:
                        val = float(val_str)
                    except ValueError:
                        val = val_str
                interventions[var] = val

            # Natural language interventions: 'setting X to 1', 'if X is set to 1', 'if X had been 1'
            nl_matches = re.finditer(
                r"\b(?:setting|set|intervene\s+on)\s+([A-Za-z_][A-Za-z0-9_]*)\s+(?:to|=)\s*([A-Za-z0-9_.-]+)",
                text,
                re.IGNORECASE,
            )
            for m in nl_matches:
                var = m.group(1).strip()
                val_str = m.group(2).strip()
                try:
                    val = int(val_str)
                except ValueError:
                    try:
                        val = float(val_str)
                    except ValueError:
                        val = val_str
                interventions[var] = val

            if_matches = re.finditer(
                r"\bif\s+([A-Za-z_][A-Za-z0-9_]*)\s+(?:had\s+been|was|were|is)\s+(?:set\s+to\s+)?([A-Za-z0-9_.-]+)",
                text,
                re.IGNORECASE,
            )
            for m in if_matches:
                var = m.group(1).strip()
                val_str = m.group(2).strip()
                try:
                    val = int(val_str)
                except ValueError:
                    try:
                        val = float(val_str)
                    except ValueError:
                        val = val_str
                interventions[var] = val

            causes: List[str] = []
            effects: List[str] = []

            # Graph arrows and chains: X -> Y -> Z
            arrow_chains = re.findall(
                r"[A-Za-z_][A-Za-z0-9_]*(?:\s*-(?:->|>)\s*[A-Za-z_][A-Za-z0-9_]*)+",
                text,
            )
            for chain in arrow_chains:
                nodes = [n.strip() for n in re.split(r"-(?:->|>)", chain) if n.strip()]
                for i in range(len(nodes) - 1):
                    c, e = nodes[i], nodes[i + 1]
                    if c not in causes:
                        causes.append(c)
                    if e not in effects:
                        effects.append(e)

            # Relational verbs: 'X causes Y', 'X leads to Y', 'X affects Y'
            verb_matches = re.finditer(
                r"\b([A-Za-z_][A-Za-z0-9_]*)\s+(?:causes?|leads?\s+to|affects?|influences?)\s+([A-Za-z_][A-Za-z0-9_]*)\b",
                text,
                re.IGNORECASE,
            )
            for m in verb_matches:
                c, e = m.group(1).strip(), m.group(2).strip()
                if c.lower() not in self.STOP_WORDS and c not in causes:
                    causes.append(c)
                if e.lower() not in self.STOP_WORDS and e not in effects:
                    effects.append(e)

            # Explanatory: 'Y because X'
            because_matches = re.finditer(
                r"\b([A-Za-z_][A-Za-z0-9_]*)\s+because\s+([A-Za-z_][A-Za-z0-9_]*)\b",
                text,
                re.IGNORECASE,
            )
            for m in because_matches:
                e, c = m.group(1).strip(), m.group(2).strip()
                if c.lower() not in self.STOP_WORDS and c not in causes:
                    causes.append(c)
                if e.lower() not in self.STOP_WORDS and e not in effects:
                    effects.append(e)

            return causes, effects, interventions, is_cf

        q_causes, q_effects, q_interv, q_cf = _extract_causal_slots(question)
        has_causal_q = bool(q_causes or q_effects or q_interv or q_cf or "cause" in question.lower())

        q_claim = (
            CausalClaim(
                causes=q_causes,
                effects=q_effects,
                interventions=q_interv,
                is_counterfactual=q_cf,
            )
            if has_causal_q
            else None
        )

        choice_claims: List[CausalClaim] = []
        for choice in choices:
            c_causes, c_effects, c_interv, c_cf = _extract_causal_slots(choice)
            # Inherit interventions or counterfactual flag from question if choice does not override
            final_interv = dict(q_interv)
            final_interv.update(c_interv)
            final_cf = c_cf or q_cf

            choice_claims.append(
                CausalClaim(
                    causes=c_causes or list(q_causes),
                    effects=c_effects or list(q_effects),
                    interventions=final_interv,
                    is_counterfactual=final_cf,
                )
            )

        return q_claim, choice_claims
