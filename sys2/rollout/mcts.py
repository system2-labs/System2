"""mcts.py - Monte Carlo Tree Search & Dynamic Compute Deliberation Engine.

Implements test-time compute scaling:
- Dynamic tree search over intermediate reasoning hypotheses and candidate choices.
- UCB1 exploration-exploitation balance over candidate branches.
- Dynamic compute allocation: scales rollout budget proportionally to ambiguity or entropy.
- Grounding via external symbolic checks (logic, units, algebra) during simulation.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class DeliberationNode:
    """A search tree node in the deliberation graph."""
    state_desc: str
    parent: Optional["DeliberationNode"] = None
    children: List["DeliberationNode"] = field(default_factory=list)
    action: Optional[str] = None  # Choice index or premise text
    visits: int = 0
    value_sum: float = 0.0
    verified_valid: bool = True
    invalidation_reason: Optional[str] = None

    @property
    def q_value(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits

    def ucb1(self, exploration_constant: float = 1.414) -> float:
        if self.visits == 0:
            return float("inf")
        if not self.verified_valid:
            return -1e9  # Heavily penalize invalid branches
        parent_visits = self.parent.visits if self.parent else self.visits
        return self.q_value + exploration_constant * math.sqrt(math.log(parent_visits + 1) / self.visits)


@dataclass
class MCTSSearchResult:
    """Outcome of dynamic MCTS deliberation."""
    best_choice: str
    best_choice_idx: int
    choice_values: Dict[int, float]
    choice_visits: Dict[int, int]
    total_rollouts: int
    search_depth: int
    computation_scaled: bool
    all_invalid: bool
    audit_trace: List[str]


class MCTSDeliberationEngine:
    """Dynamic tree search allocator for test-time deliberative compute."""

    def __init__(
        self,
        base_rollouts: int = 20,
        max_rollouts: int = 100,
        exploration_constant: float = 1.414,
    ):
        self.base_rollouts = base_rollouts
        self.max_rollouts = max_rollouts
        self.c = exploration_constant

    def search(
        self,
        question: str,
        choices: List[str],
        prior_scores: Optional[List[float]] = None,
        verifier_fn: Optional[Callable[[str, str], Tuple[bool, str]]] = None,
    ) -> MCTSSearchResult:
        """Execute dynamic tree search over candidate choices.

        Allocates rollouts dynamically based on choice margin / uncertainty.
        Memoizes external verifications to prevent redundant calls.
        """
        if not choices:
            raise ValueError("Cannot perform MCTS deliberation over empty choices.")

        root = DeliberationNode(state_desc=f"Root: {question}")
        trace: List[str] = []

        # Normalize prior scores to [0, 1]
        norm_priors = None
        if prior_scores is not None and len(prior_scores) == len(choices):
            min_s = min(prior_scores)
            max_s = max(prior_scores)
            if max_s > min_s:
                norm_priors = [(s - min_s) / (max_s - min_s) for s in prior_scores]
            else:
                norm_priors = [1.0 / len(choices)] * len(choices)

        # Initialize children for candidate choices
        for idx, choice in enumerate(choices):
            child = DeliberationNode(
                state_desc=f"Hypothesis[{idx}]: {choice}",
                parent=root,
                action=str(idx),
            )
            if norm_priors is not None:
                child.value_sum = float(norm_priors[idx])
                child.visits = 1
                root.visits += 1
            root.children.append(child)

        # Dynamic compute scaling: check if top choices are ambiguous
        rollout_budget = self.base_rollouts
        if prior_scores is not None and len(prior_scores) >= 2:
            sorted_raw = sorted(prior_scores, reverse=True)
            margin = sorted_raw[0] - sorted_raw[1]
            if margin < 0.15:  # High ambiguity -> scale compute up!
                rollout_budget = self.max_rollouts
                trace.append(f"High ambiguity detected (margin {margin:.3f} < 0.15). Scaling rollouts to {rollout_budget}.")
            else:
                trace.append(f"Clear margin ({margin:.3f} >= 0.15). Using base rollouts {rollout_budget}.")
        else:
            trace.append(f"Defaulting to base rollouts: {rollout_budget}.")

        # Memoization cache for deterministic verifier checks
        verifier_cache: Dict[int, Tuple[bool, str]] = {}

        # Execute MCTS Rollout Loop
        for sim in range(rollout_budget):
            # 1. Selection
            best_node = max(root.children, key=lambda n: n.ucb1(self.c))
            choice_idx = int(best_node.action)
            candidate_text = choices[choice_idx]

            # 2. External Verification (Grounding with memoization)
            reward = 0.5
            if verifier_fn is not None:
                if choice_idx not in verifier_cache:
                    verifier_cache[choice_idx] = verifier_fn(question, candidate_text)

                is_valid, reason = verifier_cache[choice_idx]
                if not is_valid:
                    best_node.verified_valid = False
                    best_node.invalidation_reason = reason
                    reward = 0.0
                    if f"Choice {choice_idx} invalidated" not in "".join(trace):
                        trace.append(f"Sim {sim}: Choice {choice_idx} invalidated by verifier: {reason}")
                else:
                    reward = 1.0
            else:
                reward = 0.8  # neutral simulated progress

            # 3. Backpropagation
            best_node.visits += 1
            best_node.value_sum += reward
            root.visits += 1

        # Aggregation
        choice_values = {}
        choice_visits = {}
        for child in root.children:
            idx = int(child.action)
            choice_values[idx] = child.q_value
            choice_visits[idx] = child.visits

        # Pick highest visited valid choice
        valid_children = [c for c in root.children if c.verified_valid]
        all_invalid = (len(valid_children) == 0)

        if not valid_children:
            # Fall back to prior scores rather than index 0 lock
            trace.append("WARNING: All candidate choices were invalidated by verifiers!")
            if norm_priors is not None:
                selected_idx = max(range(len(choices)), key=lambda i: norm_priors[i])
            else:
                selected_idx = 0
        else:
            selected = max(valid_children, key=lambda c: (c.visits, c.q_value))
            selected_idx = int(selected.action)

        return MCTSSearchResult(
            best_choice=choices[selected_idx],
            best_choice_idx=selected_idx,
            choice_values=choice_values,
            choice_visits=choice_visits,
            total_rollouts=root.visits,
            search_depth=2,
            computation_scaled=(rollout_budget > self.base_rollouts),
            all_invalid=all_invalid,
            audit_trace=trace,
        )
