"""registry.py - Unified System2 Weight Registry and Integrity Verification.

Provides typed, verified access to all specialized track checkpoints:
- Track A (Science): models/system2.pt
- Track B (Code): models/system2_code.pt
- Track C (Medicine): models/system2_med.pt
- Track D (Electrical): models/system2_electrical.pt
"""

import os
import json
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import torch

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MANIFEST_PATH = os.path.join(ROOT_DIR, "weights_manifest.json")


@dataclass
class TrackWeightMeta:
    id: str
    name: str
    domain: str
    checkpoint: str
    file_size_mb: float
    sha256: str
    guarded_metric: float
    accuracy: float
    brier: float
    calibration_tau: Optional[float]
    lora_params: int
    thought_head_params: int
    judger_params: int
    total_specialized_params: int
    dataset_scope: str = ""
    held_out_eval: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


class WeightRegistry:
    """Single source of truth for System2 specialized weight checkpoints and metadata."""

    def __init__(self, manifest_data: Dict[str, Any], manifest_path: str):
        self.manifest_path = manifest_path
        self.version = manifest_data.get("version", "1.0.0")
        self.base_model = manifest_data.get("base_model", {})
        self.subspace_metrics = manifest_data.get("subspace_metrics", {})
        self._tracks: Dict[str, TrackWeightMeta] = {}

        for t in manifest_data.get("tracks", []):
            meta = TrackWeightMeta(
                id=t["id"],
                name=t["name"],
                domain=t["domain"],
                checkpoint=t["checkpoint"],
                file_size_mb=t.get("file_size_mb", 0.0),
                sha256=t.get("sha256", ""),
                guarded_metric=t.get("guarded_metric", 0.0),
                accuracy=t.get("accuracy", 0.0),
                brier=t.get("brier", 0.0),
                calibration_tau=t.get("calibration_tau"),
                lora_params=t.get("lora_params", 0),
                thought_head_params=t.get("thought_head_params", 0),
                judger_params=t.get("judger_params", 0),
                total_specialized_params=t.get("total_specialized_params", 0),
                dataset_scope=t.get("dataset_scope", ""),
                held_out_eval=t.get("held_out_eval", ""),
                extra=t,
            )
            self._tracks[meta.id] = meta

    @classmethod
    def load(cls, manifest_path: Optional[str] = None) -> "WeightRegistry":
        path = manifest_path or DEFAULT_MANIFEST_PATH
        if not os.path.exists(path):
            raise FileNotFoundError(f"Manifest not found: {path}. Run scripts/map_specialized_weights.py first.")
        with open(path, "r") as f:
            data = json.load(f)
        return cls(data, path)

    def list_tracks(self) -> List[TrackWeightMeta]:
        return list(self._tracks.values())

    def get_track(self, track_id: str) -> TrackWeightMeta:
        if track_id not in self._tracks:
            raise KeyError(f"Track '{track_id}' not found in registry. Available: {list(self._tracks.keys())}")
        return self._tracks[track_id]

    def verify_integrity(self, track_id: str) -> bool:
        meta = self.get_track(track_id)
        if not os.path.exists(meta.checkpoint):
            return False
        sha = hashlib.sha256()
        with open(meta.checkpoint, "rb") as f:
            while chunk := f.read(1024 * 1024):
                sha.update(chunk)
        return sha.hexdigest() == meta.sha256

    def load_track_weights(self, track_id: str, device: str = "cpu") -> Dict[str, Any]:
        meta = self.get_track(track_id)
        return torch.load(meta.checkpoint, map_location=device, weights_only=False)

    def get_subspace_similarity_matrix(self) -> Dict[str, float]:
        return self.subspace_metrics.get("overall_lora_similarities", {})

    def to_summary_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "base_model": self.base_model,
            "tracks": [
                {
                    "id": t.id,
                    "name": t.name,
                    "domain": t.domain,
                    "checkpoint": t.checkpoint,
                    "file_size_mb": t.file_size_mb,
                    "sha256": t.sha256,
                    "guarded_metric": t.guarded_metric,
                    "accuracy": t.accuracy,
                    "brier": t.brier,
                    "calibration_tau": t.calibration_tau,
                    "total_specialized_params": t.total_specialized_params,
                    "lora_params": t.lora_params,
                    "thought_head_params": t.thought_head_params,
                    "judger_params": t.judger_params,
                }
                for t in self.list_tracks()
            ],
            "subspace_metrics": self.subspace_metrics,
        }
