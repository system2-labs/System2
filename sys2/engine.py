"""engine.py - Unified System2 Deliberation Engine with Dynamic LoRA Adapter Hot-Swapping.

Maintains a single frozen ModernBERT-Large (395M) backbone resident in memory and
dynamically hot-swaps the 1.57MB LoRA + 5-head transformer stack + Judger weights
in < 3ms based on the prompt's routed domain.
"""

import os
import re
import sys
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

os.environ["TRANSFORMERS_VERBOSITY"] = "error"

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from bert import DEFAULT_BACKBONE, get_bert  # noqa: E402
from encoding import mean_pool_pairwise  # noqa: E402
from load_system2 import extract_config  # noqa: E402
from lora import apply_lora  # noqa: E402
from routing.domain_router import DomainRouter  # noqa: E402
from system import System2  # noqa: E402

AUTHORITY_PREAMBLE_PATTERN = re.compile(
    r"^(?:Note:\s*)?(?:(?:Leading|Senior|Domain|Expert|Unanimous)\s+)?(?:domain\s+)?"
    r"(?:authorities|reviewers|experts|scientists|analysts|consensus)\s+"
    r"(?:unanimously\s+)?(?:concluded|confirmed|asserted|agreed|stated|claimed)\s+"
    r"(?:that\s+the\s+answer\s+directly\s+entails:?|that\s+option\s+\(?[A-D]\)?\s+is\s+correct:?|that:?|:)\s*[^.\n]+[.\n]\s*",
    re.IGNORECASE,
)


def sanitize_authority_preamble(text: str) -> str:
    """Strip synthetic authority injection preambles before neural encoding."""
    cleaned = AUTHORITY_PREAMBLE_PATTERN.sub("", text).strip()
    return cleaned if cleaned else text

CHECKPOINT_MAP = {
    "science": os.path.join(ROOT_DIR, "models", "system2.pt"),
    "code": os.path.join(ROOT_DIR, "models", "system2_code.pt"),
    "med": os.path.join(ROOT_DIR, "models", "system2_med.pt"),
    "electrical": os.path.join(ROOT_DIR, "models", "system2_electrical.pt"),
    "auto": os.path.join(ROOT_DIR, "models", "system2_auto.pt"),
    "culinary": os.path.join(ROOT_DIR, "models", "system2_culinary.pt"),
    "math": os.path.join(ROOT_DIR, "models", "system2_math.pt"),
    "legal": os.path.join(ROOT_DIR, "models", "system2_legal.pt"),
    "finance": os.path.join(ROOT_DIR, "models", "system2_finance.pt"),
    "physics": os.path.join(ROOT_DIR, "models", "system2_physics.pt"),
    "safety": os.path.join(ROOT_DIR, "models", "system2_safety.pt"),
    "eval": os.path.join(ROOT_DIR, "models", "system2_eval.pt"),
    "logic": os.path.join(ROOT_DIR, "models", "system2_logic.pt"),
}


@dataclass
class DeliberationResult:
    selected_index: int
    selected_choice: str
    confidence: float
    head_consensus: str  # e.g. "4/5"
    per_head_choices: List[int]
    domain: str
    routing_method: str
    routing_latency_ms: float
    deliberation_latency_ms: float
    total_latency_ms: float
    choice_probabilities: Optional[List[float]] = None

    def __str__(self) -> str:
        letters = ["A", "B", "C", "D"]
        let = letters[self.selected_index] if 0 <= self.selected_index < 4 else f"#{self.selected_index}"
        return (
            f"Verdict: ({let}) {self.selected_choice}\n"
            f"Confidence: {self.confidence * 100:.1f}% | Consensus: {self.head_consensus} Heads\n"
            f"Domain: [{self.domain.upper()}] ({self.routing_method}, route={self.routing_latency_ms:.2f}ms, delib={self.deliberation_latency_ms:.2f}ms, total={self.total_latency_ms:.2f}ms)"
        )


class System2Engine:
    """Unified Multi-Domain System 2 Deliberation Engine with Hot-Swapping."""

    def __init__(self, device: Optional[str] = None):
        if device is None:
            self.device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self._lock = threading.RLock()
        print(f"[System2Engine] Initializing on {self.device.upper()}...", file=sys.stderr)
        # Deferred from module level: importing transformers costs seconds, and the
        # env var above is not re-read if something imported it first.
        import transformers
        transformers.logging.set_verbosity_error()

        # One resident backbone for the whole process: bert.get_bert() owns the cache,
        # and the router below reuses this exact object instead of loading its own.
        self.tokenizer, self.model = get_bert(DEFAULT_BACKBONE)
        self.model.to(self.device)
        self.model.eval()

        # Inject LoRA scaffolding into top 8 layers
        apply_lora(self.model, r=32, alpha=32, num_layers=8)

        # Dynamic Router: centroids live on CPU, but the backbone is shared with this
        # engine (and its LoRA layers are bypassed during routing via disable_lora).
        self.router = DomainRouter(device="cpu", bert_model=self.model, tokenizer=self.tokenizer)

        # In-memory adapter cache: domain -> (encoder_state, sys2_instance)
        self.adapters: Dict[str, Tuple[Dict[str, torch.Tensor], System2]] = {}
        self.current_domain: Optional[str] = None
        self.current_sys2: Optional[System2] = None

    def _load_adapter(self, domain: str) -> Tuple[Dict[str, torch.Tensor], System2]:
        """Load and cache domain specialized weights."""
        if domain in self.adapters:
            return self.adapters[domain]

        ckpt_path = CHECKPOINT_MAP.get(domain, CHECKPOINT_MAP["science"])
        if not os.path.exists(ckpt_path):
            fallback = CHECKPOINT_MAP["science"]
            print(f"[System2Engine warning] Checkpoint {ckpt_path} missing, falling back to {fallback}", file=sys.stderr)
            ckpt_path = fallback

        # weights_only=True safely deserializes tensor dicts plus the primitive/list/dict
        # metadata keys consumed by extract_config across all domain checkpoints without
        # executing arbitrary pickle code.
        state = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        config, sys2_state = extract_config(state)

        # Extract encoder LoRA parameters (and merge trained LayerNorms if saved).
        # Shallow-copy before merging so we never mutate the deserialized checkpoint dict.
        encoder_state: Dict[str, torch.Tensor] = {}
        if isinstance(state, dict) and isinstance(state.get("lora"), dict):
            encoder_state = dict(state["lora"])
        elif isinstance(state, dict) and isinstance(state.get("encoder"), dict):
            encoder_state = dict(state["encoder"])
        elif isinstance(state, dict):
            encoder_state = {k: v for k, v in state.items() if "lora_" in k or "layers." in k}
        if isinstance(state, dict) and isinstance(state.get("lora_norms"), dict):
            encoder_state.update(state["lora_norms"])

        # Reconstruct System2 from checkpoint metadata/shapes rather than hardcoded defaults
        # so per-domain ensemble_temp, head_weights, judger_soft_emb, and learnable calibration
        # tensors (e.g. log_decision_tau) are never silently overridden or dropped.
        sys2 = System2(**config)
        try:
            sys2.load_state_dict(sys2_state, strict=True)
        except RuntimeError as e:
            print(
                f"[System2Engine warning] Strict load failed for {ckpt_path} ({e}), retrying with strict=False...",
                file=sys.stderr,
            )
            sys2.load_state_dict(sys2_state, strict=False)
        sys2.to(self.device)
        sys2.eval()

        self.adapters[domain] = (encoder_state, sys2)
        return encoder_state, sys2

    def switch_domain(self, domain: str) -> None:
        """Hot-swap the domain adapter in < 3ms with thread-safe locking."""
        with self._lock:
            if self.current_domain == domain and self.current_sys2 is not None:
                return

            encoder_state, sys2 = self._load_adapter(domain)
            if encoder_state:
                self.model.load_state_dict(encoder_state, strict=False)
            else:
                # Reset LoRA B parameters to zero so previous domain adapter doesn't leak
                with torch.no_grad():
                    for name, param in self.model.named_parameters():
                        if "lora_B" in name:
                            param.zero_()
            self.current_sys2 = sys2
            self.current_domain = domain

    def adjudicate(
        self,
        question: str,
        choices: List[str],
        domain: Optional[str] = None,
        max_len: int = 1024,
    ) -> DeliberationResult:
        """Adjudicate candidate choices in < 35ms through 5-head parallel deliberation."""
        if not choices:
            raise ValueError("choices must be a non-empty list.")

        t_start = time.perf_counter()

        # Sanitize authority injection preambles before feature extraction
        clean_question = sanitize_authority_preamble(question)

        # 1. Dynamic Routing
        t_route_0 = time.perf_counter()
        if domain is None:
            # Serialize routing with adapter switching and forward passes on the
            # shared backbone (self._lock is re-entrant).
            with self._lock:
                route_decision = self.router.route(clean_question)
            selected_domain = route_decision.selected_track
            routing_method = route_decision.method
        else:
            selected_domain = domain
            routing_method = "manual"
        t_route_ms = (time.perf_counter() - t_route_0) * 1000

        # 2. Hot-Swap Adapter & Deliberation Forward Pass (Thread-Safe)
        with self._lock:
            self.switch_domain(selected_domain)

            num_raw = len(choices)
            if num_raw < 4:
                # Pad <4-choice queries to 4 slots to match the 4-choice geometry checkpoints
                # were trained on; choice_mask hides padding from attention, centroid contrast,
                # per-head normalization, and voting.
                eval_choices = list(choices) + ["N/A"] * (4 - num_raw)
                choice_mask = torch.zeros((1, 4), dtype=torch.bool, device=self.device)
                choice_mask[:, :num_raw] = True
            else:
                eval_choices = list(choices)
                choice_mask = None

            # 3. Deliberation Forward Pass
            t_delib_0 = time.perf_counter()
            with torch.no_grad():
                vecs = mean_pool_pairwise(
                    [clean_question],
                    [eval_choices],
                    max_len=max_len,
                    device=self.device,
                    with_grad=False,
                    asym_spans=True,
                    asym_span_mode="choice_forward",
                    model=self.model,
                    tokenizer=self.tokenizer,
                )
                best, conf, pruned, confs, logits, emb, scores, ens_logits = self.current_sys2(
                    vecs, choice_mask=choice_mask
                )
                t_delib_ms = (time.perf_counter() - t_delib_0) * 1000

                valid_ens = ens_logits[0, :num_raw]
                win_idx = int(valid_ens.argmax(dim=-1).item())
                raw_probs = torch.softmax(valid_ens, dim=-1).tolist()

                if isinstance(raw_probs, float):
                    raw_probs = [raw_probs]
                choice_probs = [round(float(p), 4) for p in raw_probs]

                # System2.forward sets padded logits to -1e9 so argmax votes strictly over 0..num_raw-1.
                head_preds = logits.squeeze(0).argmax(dim=-1).tolist()
                consensus = head_preds.count(win_idx)
                calibrated_conf = conf.item() if hasattr(conf, "item") else float(conf)

                # Non-Parametric Epistemic Grounding:
                # If the consensus is severely fractured (<= 2/5 heads) or raw logit support is depressed,
                # scale decision confidence to prevent overconfident hallucinations.
                if consensus <= 2:
                    calibrated_conf = min(calibrated_conf, 0.40)

        t_total_ms = (time.perf_counter() - t_start) * 1000

        return DeliberationResult(
            selected_index=win_idx,
            selected_choice=choices[win_idx] if 0 <= win_idx < len(choices) else "",
            confidence=round(calibrated_conf, 4),
            head_consensus=f"{consensus}/5",
            per_head_choices=head_preds,
            domain=selected_domain,
            routing_method=routing_method,
            routing_latency_ms=round(t_route_ms, 2),
            deliberation_latency_ms=round(t_delib_ms, 2),
            total_latency_ms=round(t_total_ms, 2),
            choice_probabilities=choice_probs,
        )
