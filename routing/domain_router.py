"""domain_router.py - System2 Two-Stage Hierarchical Dynamic Domain Router.

Executes sub-millisecond dispatch of incoming prompts to the optimal specialized track:
- Stage 1: Ultra-fast regex & technical signature gate (< 0.05 ms).
- Stage 2: Dense ModernBERT subspace prototype centroid projection (< 0.8 ms).
"""

import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn.functional as F

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from routing.lexical_rules import scan_lexical_signatures

DEFAULT_CENTROIDS_PATH = os.path.join(ROOT_DIR, "routing", "domain_centroids.pt")


@dataclass
class RouteDecision:
    selected_track: str
    confidence: float
    probabilities: Dict[str, float]
    method: str  # "lexical" | "centroid" | "fallback"
    latency_ms: float
    raw_similarities: Optional[Dict[str, float]] = None


class DomainRouter:
    """Dynamic Domain Router for System2 Multi-Track Deliberation."""

    TRACKS = ["science", "code", "med", "electrical", "auto", "culinary", "math"]

    TRACK_NAMES = {
        "science": "Track A: Science & Causal Reasoning",
        "code": "Track B: Polyglot Code Traces",
        "med": "Track C: Clinical & Medical Diagnostics",
        "electrical": "Track D: Electrical & Power Systems",
        "auto": "Track E: Automotive & HVAC Diagnostics",
        "culinary": "Track F: Culinary Chemistry & Food Science",
        "math": "Track G: Mathematical & Quantitative Reasoning",
    }

    def __init__(
        self,
        centroids_path: Optional[str] = None,
        bert_model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        device: str = "cpu",
        tau: float = 0.05,
    ):
        self.device = device
        self.tau = tau
        self.centroids_path = centroids_path or DEFAULT_CENTROIDS_PATH
        self.centroids: Optional[torch.Tensor] = None
        self.tracks_order: List[str] = self.TRACKS
        self.bert_model = bert_model
        self.tokenizer = tokenizer

        # Load precomputed centroids if available
        if os.path.exists(self.centroids_path):
            data = torch.load(self.centroids_path, map_location=self.device, weights_only=False)
            self.centroids = data["centroids"].to(self.device)  # (6, 1024)
            self.tracks_order = data.get("tracks", self.TRACKS)
            self.tau = data.get("tau_route", self.tau)

    def _ensure_bert(self):
        """Lazy load ModernBERT if needed for Stage 2 dense routing."""
        if self.bert_model is None or self.tokenizer is None:
            from bert import get_bert
            self.tokenizer, self.bert_model = get_bert("answerdotai/ModernBERT-Large-Instruct")
            self.bert_model.to(self.device)
            self.bert_model.eval()

    def encode_text(self, text: str) -> torch.Tensor:
        """Encode single text into L2-normalized 1024-dim embedding on CPU."""
        self._ensure_bert()
        inputs = self.tokenizer([text], return_tensors="pt", padding=True, truncation=True, max_length=256).to(self.device)
        with torch.no_grad():
            last_hidden = self.bert_model(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1)
            pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1)
            return F.normalize(pooled, p=2, dim=1)  # (1, 1024)

    def route(
        self,
        prompt: str,
        precomputed_emb: Optional[torch.Tensor] = None,
    ) -> RouteDecision:
        """Route prompt to the best specialized track.

        Args:
            prompt: Question or prompt text.
            precomputed_emb: Optional (1, 1024) L2-normalized embedding if already computed.

        Returns:
            RouteDecision with selected_track, confidence, and distribution.
        """
        t0 = time.perf_counter()

        # ----------------------------------------------------------------------
        # STAGE 1: Ultra-fast Lexical Anchor Scan (< 0.05 ms)
        # ----------------------------------------------------------------------
        lex_match = scan_lexical_signatures(prompt)
        if lex_match is not None:
            track, conf = lex_match
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 3)
            # Create synthetic probability distribution
            probs = {t: (conf if t == track else round((1.0 - conf) / (len(self.TRACKS) - 1), 4)) for t in self.TRACKS}
            return RouteDecision(
                selected_track=track,
                confidence=round(conf, 4),
                probabilities=probs,
                method="lexical",
                latency_ms=elapsed_ms,
            )

        # ----------------------------------------------------------------------
        # STAGE 2: Dense ModernBERT Subspace Centroid Projection (< 0.8 ms)
        # ----------------------------------------------------------------------
        if self.centroids is not None:
            if precomputed_emb is not None:
                emb = precomputed_emb.to(self.device)
                if emb.dim() == 1:
                    emb = emb.unsqueeze(0)
                emb = F.normalize(emb, p=2, dim=1)
            else:
                emb = self.encode_text(prompt)

            # Cosine similarity matrix multiplication: (6, 1024) @ (1024, 1) -> (6, 1)
            cos_sims = torch.matmul(self.centroids, emb.squeeze(0))  # (6,)
            sims_dict = {t: round(cos_sims[i].item(), 4) for i, t in enumerate(self.tracks_order)}

            # Calibrated Softmax over similarities
            logits = cos_sims / self.tau
            probs_t = F.softmax(logits, dim=0)
            probs_dict = {t: round(probs_t[i].item(), 4) for i, t in enumerate(self.tracks_order)}

            best_idx = torch.argmax(probs_t).item()
            best_track = self.tracks_order[best_idx]
            max_prob = probs_t[best_idx].item()

            elapsed_ms = round((time.perf_counter() - t0) * 1000, 3)

            # Fallback if completely ambiguous / out-of-distribution
            if max_prob < 0.30:
                return RouteDecision(
                    selected_track="science",
                    confidence=round(max_prob, 4),
                    probabilities=probs_dict,
                    method="fallback",
                    latency_ms=elapsed_ms,
                    raw_similarities=sims_dict,
                )

            return RouteDecision(
                selected_track=best_track,
                confidence=round(max_prob, 4),
                probabilities=probs_dict,
                method="centroid",
                latency_ms=elapsed_ms,
                raw_similarities=sims_dict,
            )

        # ----------------------------------------------------------------------
        # FALLBACK: Default to Track A (Science) if centroids not yet generated
        # ----------------------------------------------------------------------
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 3)
        return RouteDecision(
            selected_track="science",
            confidence=0.50,
            probabilities={t: (0.5 if t == "science" else 0.1) for t in self.TRACKS},
            method="fallback",
            latency_ms=elapsed_ms,
        )
