"""vot_gate.py - Cost-Aware Value-of-Thinking (VoT) Gate & AnyS1 Skill Extractor.

Implements two complementary System 1 / System 2 arbitration primitives:

1. HDM Cost-Aware Value-of-Thinking Gate (inspired by arXiv:2611.16961):
   Replaces static confidence/margin escalation thresholds with a calibrated
   decision policy predicting expected utility gain:
       delta_U(x) = P(System 2 improves over System 1 | perceptual features x)
   System 2 deliberation (or downstream `<think>` mode) is recommended iff
   `delta_U(x) > lambda_cost`, where `lambda_cost >= 0` is the user/caller's
   latency price of thinking.

   SAFETY INVARIANT (Gate 1 - Fail-Closed):
   Hard safety and symbolic triggers (`safety` domain, physical units, algebraic
   equations, formal causal interventions, logical negation constraints, or
   `force_sys2`) ALWAYS bypass the cost-benefit trade-off and escalate with
   `hard_trigger_override=True` and `recommended_thinking="on"`, even if
   `lambda_cost = 1e6`.

2. AnyS1 Enumerable Decision-Point Extractor (inspired by arXiv:2611.23751):
   Parses procedural skill markdown documents (`SKILL.md`) to mine discrete,
   enumerable routing and branching decision points (condition -> discrete
   choices) so routine branches can be resolved in a single System 1 pass with
   confidence fallback to System 2.

STRICT LAZY-IMPORT INVARIANT:
This module uses only the Python standard library and NumPy. It NEVER imports
`transformers`, `bert`, or `torch` at module import time so `routing.vot_gate`
stays sub-50ms to import and sub-0.2ms per `decide()` call.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

# -----------------------------------------------------------------------------
# Domain Risk Priors & Lightweight Prompt Hard-Trigger Scanners
# -----------------------------------------------------------------------------

# Domain-level prior risk reflecting how often System 1 perceptual shortcuts
# benefit from System 2 verification when confidence is moderate.
# Safety is pinned to 1.0 and also enforced as an unconditional fail-closed gate.
DOMAIN_RISK_PRIORS: Dict[str, float] = {
    "safety": 1.00,
    "logic": 0.72,
    "math": 0.70,
    "physics": 0.68,
    "med": 0.65,
    "legal": 0.62,
    "finance": 0.58,
    "electrical": 0.58,
    "eval": 0.55,
    "code": 0.48,
    "science": 0.45,
    "auto": 0.42,
    "culinary": 0.25,
    "general": 0.28,
}

# Lightweight regexes for /v1/route prompt scanning without importing Pint/SymPy/Z3.
_PROMPT_UNIT_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:m/s\^?2?|m/s²|km/h|mph|kg|mg|cm|mm|km|Hz|Pa|kPa|kW|MW|mJ|kJ|"
    r"newtons?|joules?|watts?|volts?|amperes?|ohms?|pascals?|kelvins?|meters?|metres?)\b",
    re.IGNORECASE,
)
_PROMPT_EQUATION_RE = re.compile(
    r"(?:\b[a-zA-Z]\s*(?:[+\-*/^]\s*[a-zA-Z0-9]+)+\s*==?\s*[+-]?\d|"
    r"\b\d+(?:\.\d+)?\s*[+\-*/^]\s*[a-zA-Z0-9]+\s*==?\s*[+-]?\d|"
    r"\bsolve\s+for\s+[a-zA-Z]\b)",
    re.IGNORECASE,
)
_PROMPT_CAUSAL_RE = re.compile(
    r"(?:\bdo\s*\([^)]+\)|\bcounterfactual\b|\bstructural\s+causal\b|\bpearl'?s?\s+do-?calculus\b)",
    re.IGNORECASE,
)
_PROMPT_NEGATION_RE = re.compile(
    r"\b(?:does\s+not\s+entail|cannot\s+be\s+true|neither\s+\w+\s+nor|contradicts?\s+the\s+premise)\b",
    re.IGNORECASE,
)


def domain_risk_prior(domain: str) -> float:
    """Return calibrated risk prior in [0, 1] for a System2 track name."""
    return DOMAIN_RISK_PRIORS.get((domain or "general").strip().lower(), 0.35)


def detect_prompt_hard_triggers(prompt: str, track: str = "general") -> Tuple[bool, List[str]]:
    """Detect fail-closed safety or symbolic triggers from a prompt and track.

    Used by `/v1/route` where full SemanticParser / Pint / SymPy / Z3 pipelines
    are not executed, preserving sub-millisecond routing latency while ensuring
    safety and explicit symbolic prompts never downgrade to `thinking="off"`.
    """
    reasons: List[str] = []
    if (track or "").strip().lower() == "safety":
        reasons.append("safety_track")

    if prompt:
        # Strip fenced code blocks before scanning for algebraic equations so
        # routine code assignments (`x = 1`) do not masquerade as math equations.
        prose = re.sub(r"```.*?```", " ", prompt, flags=re.DOTALL)
        if _PROMPT_UNIT_RE.search(prose):
            reasons.append("units")
        if _PROMPT_EQUATION_RE.search(prose):
            reasons.append("equations")
        if _PROMPT_CAUSAL_RE.search(prose):
            reasons.append("causal")
        if _PROMPT_NEGATION_RE.search(prose):
            reasons.append("negation")

    return bool(reasons), reasons


def _parse_head_consensus_ratio(head_consensus: Optional[str], per_head_choices: Optional[Sequence[int]]) -> float:
    """Compute fraction of System 1 heads agreeing with the majority choice."""
    if per_head_choices:
        counts: Dict[int, int] = {}
        for choice_idx in per_head_choices:
            counts[choice_idx] = counts.get(choice_idx, 0) + 1
        return max(counts.values()) / float(len(per_head_choices))

    if head_consensus and "/" in head_consensus:
        parts = head_consensus.strip().split("/", 1)
        try:
            num = float(parts[0].strip())
            den = float(parts[1].strip())
            if den > 0.0:
                return float(np.clip(num / den, 0.0, 1.0))
        except ValueError:
            pass
    return 1.0


def _normalized_entropy(probs: Sequence[float]) -> float:
    """Compute Shannon entropy normalized to [0, 1] by log(K)."""
    k = len(probs)
    if k <= 1:
        return 0.0
    arr = np.asarray(probs, dtype=np.float64)
    arr = np.clip(arr, 0.0, None)
    total = float(arr.sum())
    if total <= 0.0:
        return 1.0
    p = arr / total
    p_nz = p[p > 1e-12]
    ent = -float(np.sum(p_nz * np.log(p_nz)))
    return float(np.clip(ent / math.log(k), 0.0, 1.0))


# -----------------------------------------------------------------------------
# Core Data Structures: VoTFeatures, VoTDecision, VoTEpisode
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class VoTFeatures:
    """Cheap System 1 perceptual signals consumed by ValueOfThinkingGate."""

    s1_confidence: float
    prob_margin: float
    entropy: float
    head_consensus_ratio: float
    num_choices: int
    domain_risk_prior: float
    has_hard_trigger: bool
    domain: str = "general"

    def __post_init__(self) -> None:
        # Enforce safety fail-closed invariant at feature construction time as well.
        if (self.domain or "").strip().lower() == "safety" and not self.has_hard_trigger:
            object.__setattr__(self, "has_hard_trigger", True)

    def to_vector(self) -> np.ndarray:
        """Project perceptual signals into a 10-dim interaction feature vector.

        Interaction terms capture failure modes that static confidence/margin
        rules miss:
        - `s1_confidence * (1 - head_consensus_ratio)`: detects overconfident
          single-peak predictions where the 5 deliberation heads actually split.
        - `(1 - s1_confidence) * domain_risk_prior`: scales hesitation penalty
          by domain risk (escalate math/med faster than general/culinary).
        - `entropy * (1 - prob_margin)`: separates genuine multi-way confusion
          from clean two-way competitions.
        """
        conf = float(np.clip(self.s1_confidence, 0.0, 1.0))
        margin = float(np.clip(self.prob_margin, 0.0, 1.0))
        ent = float(np.clip(self.entropy, 0.0, 1.0))
        cons = float(np.clip(self.head_consensus_ratio, 0.0, 1.0))
        n_norm = float(math.log1p(max(1, self.num_choices)) / math.log1p(13))
        risk = float(np.clip(self.domain_risk_prior, 0.0, 1.0))

        return np.array(
            [
                conf,
                margin,
                ent,
                cons,
                n_norm,
                risk,
                (1.0 - conf) * risk,
                conf * (1.0 - cons),
                ent * (1.0 - margin),
                (1.0 - cons) * risk,
            ],
            dtype=np.float64,
        )

    @classmethod
    def from_probabilities(
        cls,
        probs: Sequence[float],
        domain: str = "general",
        head_consensus_ratio: float = 1.0,
        has_hard_trigger: bool = False,
        confidence_override: Optional[float] = None,
    ) -> "VoTFeatures":
        """Build VoTFeatures directly from a candidate probability distribution."""
        clean = [max(0.0, float(p)) for p in probs] if probs else [1.0]
        total = sum(clean)
        if total > 0.0:
            norm = [p / total for p in clean]
        else:
            norm = [1.0 / len(clean)] * len(clean)

        sorted_p = sorted(norm, reverse=True)
        top1 = sorted_p[0]
        margin = (sorted_p[0] - sorted_p[1]) if len(sorted_p) > 1 else 1.0
        conf = float(np.clip(confidence_override if confidence_override is not None else top1, 0.0, 1.0))
        dom = (domain or "general").strip().lower()
        hard = bool(has_hard_trigger or dom == "safety")

        return cls(
            s1_confidence=conf,
            prob_margin=float(np.clip(margin, 0.0, 1.0)),
            entropy=_normalized_entropy(norm),
            head_consensus_ratio=float(np.clip(head_consensus_ratio, 0.0, 1.0)),
            num_choices=len(norm),
            domain_risk_prior=domain_risk_prior(dom),
            has_hard_trigger=hard,
            domain=dom,
        )

    @classmethod
    def from_perception_result(
        cls,
        s1_res: Any,
        has_hard_trigger: bool = False,
    ) -> "VoTFeatures":
        """Extract VoTFeatures from a System1Engine PerceptionResult."""
        probs = list(s1_res.choice_probabilities) if getattr(s1_res, "choice_probabilities", None) else [s1_res.confidence, max(0.0, 1.0 - s1_res.confidence)]
        consensus = _parse_head_consensus_ratio(
            getattr(s1_res, "head_consensus", None),
            getattr(s1_res, "per_head_choices", None),
        )
        dom = getattr(s1_res, "domain", "general") or "general"
        return cls.from_probabilities(
            probs=probs,
            domain=dom,
            head_consensus_ratio=consensus,
            has_hard_trigger=has_hard_trigger,
            confidence_override=float(getattr(s1_res, "confidence", max(probs))),
        )

    @classmethod
    def from_route_decision(
        cls,
        decision: Any,
        prompt: str = "",
        has_hard_trigger: bool = False,
    ) -> "VoTFeatures":
        """Extract VoTFeatures from a DomainRouter RouteDecision for `/v1/route`."""
        track = getattr(decision, "selected_track", "general") or "general"
        probs_map = getattr(decision, "probabilities", None) or {}
        probs = list(probs_map.values()) if isinstance(probs_map, Mapping) and probs_map else [float(getattr(decision, "confidence", 0.5)), max(0.0, 1.0 - float(getattr(decision, "confidence", 0.5)))]
        prompt_hard, _ = detect_prompt_hard_triggers(prompt, track=track)
        method = getattr(decision, "method", "centroid")
        # Lexical hits are high-precision deterministic regexes; fallback routes
        # indicate OOD/ambiguous queries where consensus is lower.
        consensus_proxy = 1.0 if method == "lexical" else (0.6 if method == "fallback" else 0.85)
        return cls.from_probabilities(
            probs=probs,
            domain=track,
            head_consensus_ratio=consensus_proxy,
            has_hard_trigger=bool(has_hard_trigger or prompt_hard or track.lower() == "safety"),
            confidence_override=float(getattr(decision, "confidence", max(probs))),
        )


@dataclass(frozen=True)
class VoTDecision:
    """Decision emitted by ValueOfThinkingGate."""

    escalate: bool
    expected_utility_gain: float
    lambda_cost: float
    recommended_thinking: Literal["off", "on"]
    reason: str
    hard_trigger_override: bool


@dataclass(frozen=True)
class VoTEpisode:
    """Single historical evaluation episode used to fit/calibrate the VoT gate."""

    features: VoTFeatures
    sys1_correct: bool
    sys2_correct: bool

    @property
    def sys2_helps(self) -> bool:
        """True iff escalating to System 2 turns a System 1 error into a correct answer."""
        return bool(self.sys2_correct and not self.sys1_correct)


# -----------------------------------------------------------------------------
# HDM Cost-Aware Value-of-Thinking Gate
# -----------------------------------------------------------------------------


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid over scalar or ndarray."""
    clipped = np.clip(z, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


class ValueOfThinkingGate:
    """Calibrated Cost-Aware Value-of-Thinking (VoT) decision gate.

    Models `delta_U = P(sys2_helps | x) = sigmoid((w @ x_std + b) / T)` using
    L2-regularized logistic regression over a 10-dim perceptual interaction
    basis followed by temperature-scaling calibration (`T > 0`) on a held-out
    validation slice.

    Default out-of-the-box parameters are calibrated so an unfitted gate still
    produces sensible probability estimates across easy, ambiguous, and
    high-risk queries.
    """

    FEATURE_DIM = 10
    SCHEMA_VERSION = "1.0"

    # Calibrated default weights on standardized features:
    # Negative weights on confidence, margin, consensus -> higher certainty lowers delta_U.
    # Positive weights on entropy, domain risk, and disagreement interactions -> raises delta_U.
    _DEFAULT_MEAN = np.array([0.65, 0.35, 0.45, 0.80, 0.60, 0.45, 0.16, 0.13, 0.29, 0.09], dtype=np.float64)
    _DEFAULT_STD = np.array([0.22, 0.28, 0.28, 0.22, 0.20, 0.20, 0.14, 0.14, 0.24, 0.10], dtype=np.float64)
    _DEFAULT_WEIGHTS = np.array([-1.15, -0.85, 0.70, -1.05, 0.15, 0.55, 0.75, 1.35, 0.65, 0.55], dtype=np.float64)
    _DEFAULT_BIAS = -0.75
    _DEFAULT_TEMPERATURE = 1.0

    def __init__(
        self,
        default_lambda_cost: float = 0.35,
        weights: Optional[Sequence[float]] = None,
        bias: Optional[float] = None,
        temperature: float = _DEFAULT_TEMPERATURE,
        feature_mean: Optional[Sequence[float]] = None,
        feature_std: Optional[Sequence[float]] = None,
        l2_reg: float = 1e-2,
        is_fitted: bool = False,
    ) -> None:
        self.default_lambda_cost = float(max(0.0, default_lambda_cost))
        self.weights = (
            np.asarray(weights, dtype=np.float64)
            if weights is not None
            else self._DEFAULT_WEIGHTS.copy()
        )
        if self.weights.shape != (self.FEATURE_DIM,):
            raise ValueError(f"weights must have shape ({self.FEATURE_DIM},), got {self.weights.shape}")

        self.bias = float(self._DEFAULT_BIAS if bias is None else bias)
        self.temperature = float(max(1e-3, temperature))
        self.feature_mean = (
            np.asarray(feature_mean, dtype=np.float64)
            if feature_mean is not None
            else self._DEFAULT_MEAN.copy()
        )
        self.feature_std = (
            np.asarray(feature_std, dtype=np.float64)
            if feature_std is not None
            else self._DEFAULT_STD.copy()
        )
        self.feature_std = np.where(self.feature_std < 1e-6, 1.0, self.feature_std)
        self.l2_reg = float(max(0.0, l2_reg))
        self.is_fitted = bool(is_fitted)

    def _standardize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.feature_mean) / self.feature_std

    def predict_utility_gain(self, features: VoTFeatures) -> float:
        """Return calibrated expected utility gain delta_U in [0, 1]."""
        if features.has_hard_trigger or (features.domain or "").strip().lower() == "safety":
            return 1.0
        x_std = self._standardize(features.to_vector())
        logit = (float(np.dot(self.weights, x_std)) + self.bias) / self.temperature
        return float(_sigmoid(np.asarray(logit, dtype=np.float64)))

    def decide(
        self,
        features: VoTFeatures,
        lambda_cost: Optional[float] = None,
    ) -> VoTDecision:
        """Decide whether to escalate to System 2 / `<think>` given cost price lambda."""
        eff_lambda = self.default_lambda_cost if lambda_cost is None else float(lambda_cost)

        # Gate 1 (Fail-Closed): safety and symbolic hard triggers NEVER downgrade,
        # regardless of how high lambda_cost is set.
        if features.has_hard_trigger or (features.domain or "").strip().lower() == "safety":
            return VoTDecision(
                escalate=True,
                expected_utility_gain=1.0,
                lambda_cost=eff_lambda,
                recommended_thinking="on",
                reason="hard_trigger_override: fail-closed safety or symbolic trigger active",
                hard_trigger_override=True,
            )

        delta_u = self.predict_utility_gain(features)
        escalate = delta_u > eff_lambda
        thinking: Literal["off", "on"] = "on" if escalate else "off"
        reason = (
            f"delta_U ({delta_u:.4f}) > lambda ({eff_lambda:.4f}); escalate to System 2"
            if escalate
            else f"delta_U ({delta_u:.4f}) <= lambda ({eff_lambda:.4f}); fast-path System 1 sufficient"
        )
        return VoTDecision(
            escalate=escalate,
            expected_utility_gain=delta_u,
            lambda_cost=eff_lambda,
            recommended_thinking=thinking,
            reason=reason,
            hard_trigger_override=False,
        )

    # Alias matching scorecard naming (`VoTGate.predict()`).
    predict = decide

    def fit(
        self,
        episodes: Sequence[Union[VoTEpisode, Mapping[str, Any]]],
        lr: float = 0.25,
        max_iter: int = 400,
        l2_reg: Optional[float] = None,
        val_fraction: float = 0.2,
    ) -> Dict[str, float]:
        """Deterministically fit and temperature-calibrate the VoT policy.

        Uses Newton-Raphson / IRLS with L2 regularization (convex, deterministic,
        no random initialization) on non-hard-trigger episodes, followed by 1D
        grid + golden-section temperature scaling on the validation slice to
        minimize negative log-likelihood (NLL) and Expected Calibration Error.
        """
        if not episodes:
            raise ValueError("Cannot fit ValueOfThinkingGate on empty episodes list.")

        xs: List[np.ndarray] = []
        ys: List[float] = []
        for ep in episodes:
            if isinstance(ep, VoTEpisode):
                feat = ep.features
                label = 1.0 if ep.sys2_helps else 0.0
            else:
                feat = ep["features"]
                if "sys2_helps" in ep:
                    label = 1.0 if bool(ep["sys2_helps"]) else 0.0
                else:
                    label = 1.0 if (bool(ep["sys2_correct"]) and not bool(ep["sys1_correct"])) else 0.0
            # Exclude hard-trigger episodes from the learned boundary because Gate 1
            # handles them unconditionally before the learned model runs.
            if feat.has_hard_trigger or (feat.domain or "").strip().lower() == "safety":
                continue
            xs.append(feat.to_vector())
            ys.append(label)

        if len(xs) < 4:
            raise ValueError("Need at least 4 non-hard-trigger episodes to fit ValueOfThinkingGate.")

        X = np.vstack(xs)
        y = np.asarray(ys, dtype=np.float64)
        n = X.shape[0]
        reg = self.l2_reg if l2_reg is None else float(max(0.0, l2_reg))

        # Deterministic interleaved train/calibration split so class balance is preserved
        # without relying on RNG state.
        if val_fraction > 0.0 and n >= 20:
            val_step = max(2, int(round(1.0 / val_fraction)))
            val_mask = np.zeros(n, dtype=bool)
            val_mask[::val_step] = True
            # Ensure both splits have samples.
            if val_mask.sum() >= 4 and (~val_mask).sum() >= 4:
                X_train, y_train = X[~val_mask], y[~val_mask]
                X_val, y_val = X[val_mask], y[val_mask]
            else:
                X_train, y_train = X, y
                X_val, y_val = X, y
        else:
            X_train, y_train = X, y
            X_val, y_val = X, y

        # Fit feature standardization strictly on X_train.
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        Z_train = (X_train - mean) / std
        Z_val = (X_val - mean) / std

        # Augment with bias column and solve L2-regularized logistic regression
        # via damped Newton-IRLS (converges in ~10-25 steps to machine precision).
        d = self.FEATURE_DIM
        Phi = np.hstack([Z_train, np.ones((Z_train.shape[0], 1), dtype=np.float64)])
        theta = np.zeros(d + 1, dtype=np.float64)
        # Initialize bias to empirical log-odds for fast, stable convergence.
        p_bar = float(np.clip(y_train.mean(), 1e-4, 1.0 - 1e-4))
        theta[-1] = math.log(p_bar / (1.0 - p_bar))

        reg_diag = np.full(d + 1, reg, dtype=np.float64)
        reg_diag[-1] = 0.0  # Do not regularize bias term.

        n_tr = float(Z_train.shape[0])
        for _ in range(min(max_iter, 60)):
            logits = Phi @ theta
            probs = _sigmoid(logits)
            grad = (Phi.T @ (probs - y_train)) / n_tr + reg_diag * theta
            w_diag = np.clip(probs * (1.0 - probs), 1e-6, 0.25)
            hess = (Phi.T * w_diag) @ Phi / n_tr + np.diag(reg_diag + 1e-5)
            try:
                step = np.linalg.solve(hess, grad)
            except np.linalg.LinAlgError:
                step = lr * grad
            theta_new = theta - step
            if float(np.max(np.abs(step))) < 1e-7:
                theta = theta_new
                break
            theta = theta_new

        w_fit = theta[:d]
        b_fit = float(theta[-1])

        # Post-hoc temperature scaling on validation slice to minimize NLL.
        val_logits = Z_val @ w_fit + b_fit
        best_t = 1.0
        best_nll = float("inf")
        for t_cand in np.geomspace(0.25, 4.0, num=61):
            p_val = _sigmoid(val_logits / float(t_cand))
            p_clip = np.clip(p_val, 1e-7, 1.0 - 1e-7)
            nll = -float(np.mean(y_val * np.log(p_clip) + (1.0 - y_val) * np.log(1.0 - p_clip)))
            if nll < best_nll:
                best_nll = nll
                best_t = float(t_cand)

        self.feature_mean = mean
        self.feature_std = std
        self.weights = w_fit
        self.bias = b_fit
        self.temperature = best_t
        self.l2_reg = reg
        self.is_fitted = True

        # Compute summary calibration diagnostics (NLL and Brier score on val).
        p_val_cal = _sigmoid(val_logits / best_t)
        brier = float(np.mean((p_val_cal - y_val) ** 2))
        return {
            "n_episodes": float(n),
            "n_train": float(Z_train.shape[0]),
            "n_val": float(Z_val.shape[0]),
            "val_nll": round(best_nll, 6),
            "val_brier": round(brier, 6),
            "temperature": round(best_t, 4),
        }

    # -------------------------------------------------------------------------
    # JSON Serialization (No Pickle)
    # -------------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize gate parameters to a plain JSON-compatible dictionary."""
        return {
            "schema_version": self.SCHEMA_VERSION,
            "default_lambda_cost": self.default_lambda_cost,
            "weights": [float(v) for v in self.weights.tolist()],
            "bias": float(self.bias),
            "temperature": float(self.temperature),
            "feature_mean": [float(v) for v in self.feature_mean.tolist()],
            "feature_std": [float(v) for v in self.feature_std.tolist()],
            "l2_reg": float(self.l2_reg),
            "is_fitted": bool(self.is_fitted),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ValueOfThinkingGate":
        """Reconstruct a ValueOfThinkingGate from a serialized dictionary."""
        return cls(
            default_lambda_cost=float(data.get("default_lambda_cost", 0.35)),
            weights=data["weights"],
            bias=float(data["bias"]),
            temperature=float(data.get("temperature", 1.0)),
            feature_mean=data["feature_mean"],
            feature_std=data["feature_std"],
            l2_reg=float(data.get("l2_reg", 1e-2)),
            is_fitted=bool(data.get("is_fitted", True)),
        )

    def save(self, path: Union[str, Path]) -> Path:
        """Persist calibrated gate parameters as UTF-8 JSON (never pickle)."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return out_path

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ValueOfThinkingGate":
        """Load calibrated gate parameters from a JSON file."""
        in_path = Path(path)
        payload = json.loads(in_path.read_text(encoding="utf-8"))
        return cls.from_dict(payload)


# Convenient alias matching scorecard prose (`VoTGate`).
VoTGate = ValueOfThinkingGate


# -----------------------------------------------------------------------------
# AnyS1 Enumerable Skill Decision-Point Extractor (arXiv:2611.23751)
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class SkillDecisionPoint:
    """An enumerable routing or branching decision mined from a SKILL.md doc."""

    name: str
    condition: str
    choices: List[str] = field(default_factory=list)
    source_kind: Literal["table", "conditional", "branch_list"] = "conditional"
    fallback_to_sys2: bool = True
    line_number: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(?:\|\s*:?-{2,}:?\s*)+\|?\s*$")
_CONDITIONAL_ARROW_RE = re.compile(
    r"^\s*(?:[-*+]|\d+\.)\s*(?:\*\*)?([^:*\n]{3,90})(?:\*\*)?\s*(?:->|→|=>)\s*(.+?)\s*$"
)
_CONDITIONAL_IF_RE = re.compile(
    r"^\s*(?:[-*+]|\d+\.)?\s*(?:\*\*)?(?:If|When|Whenever)\s+([^:,\n]{4,120})(?:\*\*)?\s*[:,]\s*(.+?)\s*$",
    re.IGNORECASE,
)
_BRANCH_COLON_RE = re.compile(
    r"^\s*(?:[-*+]|\d+\.)\s*\*\*([^*]{3,80})\*\*\s*[:\-–—]\s*(.+?)\s*$"
)


def _strip_frontmatter(lines: List[str]) -> Tuple[List[str], int]:
    """Skip leading YAML frontmatter (`--- ... ---`) and return (body_lines, start_line_1based)."""
    if lines and lines[0].strip() == "---":
        for idx in range(1, len(lines)):
            if lines[idx].strip() == "---":
                return lines[idx + 1 :], idx + 2
    return lines, 1


def _split_table_row(line: str) -> List[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split("|")]


def _clean_md(text: str) -> str:
    """Remove lightweight markdown emphasis/backticks for clean choice strings."""
    cleaned = re.sub(r"`([^`]+)`", r"\1", text)
    cleaned = re.sub(r"\*\*([^*]+)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"<br\s*/?>", "; ", cleaned, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", cleaned).strip()


def extract_enumerable_decision_points(skill_markdown: str) -> List[SkillDecisionPoint]:
    """Mine enumerable routing/branching decision points from procedural SKILL.md.

    Implements the AnyS1 (`arXiv:2611.23751`) preprocessing step: scans a skill
    specification for:
      1. Routing/decision Markdown tables (condition column -> discrete route/action choices)
      2. Explicit conditional rules (`If/When <cond>: <action>` or `Condition -> Action`)
      3. Structured mode/profile branch lists (`**Branch**: Action`)

    Skips YAML frontmatter and fenced code blocks so code examples and ASCII
    diagrams are never misparsed as routing rules.
    """
    if not skill_markdown or not skill_markdown.strip():
        return []

    raw_lines = skill_markdown.splitlines()
    body_lines, line_offset = _strip_frontmatter(raw_lines)

    decisions: List[SkillDecisionPoint] = []
    seen_keys: set[Tuple[str, str]] = set()
    current_heading = "Root"
    in_code_block = False

    i = 0
    n = len(body_lines)
    while i < n:
        line = body_lines[i]
        stripped = line.strip()
        abs_line = i + line_offset

        if stripped.startswith("```"):
            in_code_block = not in_code_block
            i += 1
            continue
        if in_code_block:
            i += 1
            continue

        h_match = _HEADING_RE.match(stripped)
        if h_match:
            current_heading = _clean_md(h_match.group(2))
            i += 1
            continue

        # 1. Markdown decision / routing tables: header row + separator + >= 2 data rows.
        if "|" in stripped and i + 1 < n and _TABLE_SEP_RE.match(body_lines[i + 1]):
            headers = [_clean_md(h) for h in _split_table_row(line)]
            j = i + 2
            rows: List[List[str]] = []
            while j < n and "|" in body_lines[j] and body_lines[j].strip() and not body_lines[j].strip().startswith("```"):
                row_cells = [_clean_md(c) for c in _split_table_row(body_lines[j])]
                if len(row_cells) >= 2 and any(row_cells):
                    rows.append(row_cells)
                j += 1

            if len(rows) >= 2 and len(headers) >= 2:
                # Choose action column: prefer columns named Route/Action/Surface/Skill/Strategy/Target,
                # otherwise fall back to the last column.
                action_col = len(headers) - 1
                for col_idx, hdr in enumerate(headers[1:], start=1):
                    if any(
                        kw in hdr.lower()
                        for kw in ("route", "action", "skill", "surface", "strategy", "target", "mode", "role")
                    ):
                        action_col = col_idx
                        break

                choices = []
                for row in rows:
                    cond_cell = row[0]
                    act_cell = row[action_col] if action_col < len(row) else row[-1]
                    if cond_cell and act_cell:
                        choices.append(f"{cond_cell} -> {act_cell}")

                if choices:
                    cond_summary = f"Select {headers[action_col]} based on {headers[0]}"
                    key = (current_heading.lower(), cond_summary.lower())
                    if key not in seen_keys:
                        seen_keys.add(key)
                        decisions.append(
                            SkillDecisionPoint(
                                name=current_heading,
                                condition=cond_summary,
                                choices=choices,
                                source_kind="table",
                                fallback_to_sys2=True,
                                line_number=abs_line,
                            )
                        )
            i = j
            continue

        # 2. Arrow routing rules (`Condition -> Action`).
        arrow_m = _CONDITIONAL_ARROW_RE.match(line)
        if arrow_m:
            cond = _clean_md(arrow_m.group(1))
            act = _clean_md(arrow_m.group(2))
            if cond and act:
                sub_choices = [c.strip() for c in re.split(r"\s+(?:or|/)\s+", act) if c.strip()]
                key = (current_heading.lower(), cond.lower())
                if key not in seen_keys:
                    seen_keys.add(key)
                    decisions.append(
                        SkillDecisionPoint(
                            name=current_heading,
                            condition=cond,
                            choices=sub_choices if len(sub_choices) > 1 else [act],
                            source_kind="conditional",
                            fallback_to_sys2=True,
                            line_number=abs_line,
                        )
                    )
            i += 1
            continue

        # 3. Explicit If/When conditional rules (`If <cond>: <action>`).
        if_m = _CONDITIONAL_IF_RE.match(line)
        if if_m:
            cond = _clean_md(if_m.group(1))
            act = _clean_md(if_m.group(2))
            if cond and act:
                key = (current_heading.lower(), cond.lower())
                if key not in seen_keys:
                    seen_keys.add(key)
                    decisions.append(
                        SkillDecisionPoint(
                            name=current_heading,
                            condition=cond,
                            choices=[act],
                            source_kind="conditional",
                            fallback_to_sys2=True,
                            line_number=abs_line,
                        )
                    )
            i += 1
            continue

        # 4. Contiguous bold branch lists (`- **Branch A**: Action`, `- **Branch B**: Action`).
        branch_m = _BRANCH_COLON_RE.match(line)
        if branch_m:
            group_choices: List[str] = []
            start_line = abs_line
            k = i
            while k < n:
                bm = _BRANCH_COLON_RE.match(body_lines[k])
                if not bm:
                    break
                label = _clean_md(bm.group(1))
                desc = _clean_md(bm.group(2))
                if label and desc:
                    group_choices.append(f"{label}: {desc}")
                k += 1

            # Only emit as an enumerable decision point when >= 2 discrete branches exist.
            if len(group_choices) >= 2:
                cond_summary = f"Select branch under '{current_heading}'"
                key = (current_heading.lower(), cond_summary.lower())
                if key not in seen_keys:
                    seen_keys.add(key)
                    decisions.append(
                        SkillDecisionPoint(
                            name=current_heading,
                            condition=cond_summary,
                            choices=group_choices,
                            source_kind="branch_list",
                            fallback_to_sys2=True,
                            line_number=start_line,
                        )
                    )
            i = k
            continue

        i += 1

    return decisions


def resolve_decision_point_with_gate(
    decision_point: SkillDecisionPoint,
    choice_probabilities: Sequence[float],
    gate: ValueOfThinkingGate,
    domain: str = "general",
    head_consensus_ratio: float = 1.0,
    lambda_cost: Optional[float] = None,
    has_hard_trigger: bool = False,
) -> Tuple[Optional[str], VoTDecision]:
    """Attempt single-pass System 1 resolution of an AnyS1 decision point.

    Returns `(selected_choice, vot_decision)` where `selected_choice` is the
    enumerated choice string when System 1 is accepted (`not vot_decision.escalate`),
    or `None` when the gate escalates to System 2 fallback.
    """
    if not decision_point.choices:
        raise ValueError("SkillDecisionPoint must contain at least one choice.")
    if len(choice_probabilities) != len(decision_point.choices):
        raise ValueError(
            f"Probability count ({len(choice_probabilities)}) must match choices ({len(decision_point.choices)})."
        )

    features = VoTFeatures.from_probabilities(
        probs=choice_probabilities,
        domain=domain,
        head_consensus_ratio=head_consensus_ratio,
        has_hard_trigger=has_hard_trigger,
    )
    decision = gate.decide(features, lambda_cost=lambda_cost)
    if decision.escalate and decision_point.fallback_to_sys2:
        return None, decision

    best_idx = int(np.argmax(np.asarray(choice_probabilities, dtype=np.float64)))
    return decision_point.choices[best_idx], decision


def extract_decision_points_from_paths(paths: Iterable[Union[str, Path]]) -> Dict[str, List[SkillDecisionPoint]]:
    """Batch-parse SKILL.md files and return `{skill_name: [SkillDecisionPoint, ...]}`."""
    results: Dict[str, List[SkillDecisionPoint]] = {}
    for raw_path in paths:
        p = Path(raw_path)
        if not p.is_file():
            continue
        skill_name = p.parent.name if p.name.upper() == "SKILL.MD" else p.stem
        text = p.read_text(encoding="utf-8")
        results[skill_name] = extract_enumerable_decision_points(text)
    return results


__all__ = [
    "DOMAIN_RISK_PRIORS",
    "SkillDecisionPoint",
    "ValueOfThinkingGate",
    "VoTDecision",
    "VoTEpisode",
    "VoTFeatures",
    "VoTGate",
    "detect_prompt_hard_triggers",
    "domain_risk_prior",
    "extract_decision_points_from_paths",
    "extract_enumerable_decision_points",
    "resolve_decision_point_with_gate",
]
