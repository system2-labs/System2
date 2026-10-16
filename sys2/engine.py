"""engine.py - Unified System2 Deliberation Engine with Dynamic LoRA Adapter Hot-Swapping.

Maintains a single frozen ModernBERT-Large (395M) backbone resident in memory and
dynamically hot-swaps the 1.57MB LoRA + 5-head transformer stack + Judger weights
in < 3ms based on the prompt's routed domain.
"""

import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import torch
import transformers
transformers.logging.set_verbosity_error()

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from bert import get_bert
from encoding import mean_pool_pairwise
from lora import apply_lora
from routing.domain_router import DomainRouter
from system import System2

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
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            self.device = device

        print(f"[System2Engine] Initializing on {self.device.upper()}...", file=sys.stderr)
        self.tokenizer, self.model = get_bert("answerdotai/ModernBERT-Large-Instruct")
        self.model.to(self.device)
        self.model.eval()

        # Inject LoRA scaffolding into top 8 layers
        apply_lora(self.model, r=32, alpha=32, num_layers=8)

        # Dynamic Router
        self.router = DomainRouter(device="cpu")

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

        state = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        sys2_state = state["sys2"] if isinstance(state, dict) and "sys2" in state else state

        # Extract encoder LoRA parameters if present
        encoder_state = {}
        if isinstance(state, dict) and "encoder" in state:
            encoder_state = state["encoder"]
        elif isinstance(state, dict):
            encoder_state = {k: v for k, v in state.items() if "lora_" in k or "layers." in k}

        sys2 = System2(
            num_heads=5,
            num_classes=1,
            pairwise=True,
            diff_features=True,
            head_layers=2,
            ensemble_temp=1.8,
            margin_features=True,
            judger_residual=True,
            head_norm_logits=True,
            decision_conf=True,
            decision_conf_mode="softmax",
            decision_conf_tau=0.75,
            decision_conf_alpha=0.3,
        )
        sys2.load_state_dict(sys2_state, strict=False)
        sys2.to(self.device)
        sys2.eval()

        self.adapters[domain] = (encoder_state, sys2)
        return encoder_state, sys2

    def switch_domain(self, domain: str) -> None:
        """Hot-swap the domain adapter in < 3ms."""
        if self.current_domain == domain and self.current_sys2 is not None:
            return

        encoder_state, sys2 = self._load_adapter(domain)
        if encoder_state:
            self.model.load_state_dict(encoder_state, strict=False)
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
        t_start = time.perf_counter()

        # 1. Dynamic Routing
        t_route_0 = time.perf_counter()
        if domain is None:
            route_decision = self.router.route(question)
            selected_domain = route_decision.selected_track
            routing_method = route_decision.method
        else:
            selected_domain = domain
            routing_method = "manual"
        t_route_ms = (time.perf_counter() - t_route_0) * 1000

        # 2. Hot-Swap Adapter
        self.switch_domain(selected_domain)

        # 3. Deliberation Forward Pass
        t_delib_0 = time.perf_counter()
        with torch.no_grad():
            vecs = mean_pool_pairwise(
                [question],
                [choices],
                max_len=max_len,
                device=self.device,
                with_grad=False,
                asym_spans=True,
                asym_span_mode="choice_forward",
                model=self.model,
                tokenizer=self.tokenizer,
            )
            best, conf, pruned, confs, logits, emb, scores, ens_logits = self.current_sys2(vecs)
            t_delib_ms = (time.perf_counter() - t_delib_0) * 1000

            win_idx = ens_logits.argmax(dim=-1).item()
            head_preds = logits.squeeze(0).argmax(dim=-1).tolist()
            consensus = head_preds.count(win_idx)
            calibrated_conf = conf.item() if hasattr(conf, "item") else float(conf)

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
        )
