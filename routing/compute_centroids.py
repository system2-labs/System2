#!/usr/bin/env python3
"""compute_centroids.py - Offline Prototype Centroid Generator for System2 Dynamic Domain Routing.

Extracts normalized 1024-dim ModernBERT embeddings from held-out validation/test splits
across all 6 specialized domains and computes prototype unit centroids for sub-millisecond
cosine routing.

Runs strictly on CPU to avoid GPU VRAM contention with ongoing training.
"""

import os
import sys
import torch
from datasets import load_from_disk
from typing import Dict, List

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from bert import get_bert

DATA_PATHS = {
    "science": os.path.join(ROOT_DIR, "data", "arc_augmented"),
    "code": os.path.join(ROOT_DIR, "data", "code_traces"),
    "med": os.path.join(ROOT_DIR, "data", "medical"),
    "electrical": os.path.join(ROOT_DIR, "data", "electrical"),
    "auto": os.path.join(ROOT_DIR, "data", "auto"),
    "culinary": os.path.join(ROOT_DIR, "data", "culinary"),
    "math": os.path.join(ROOT_DIR, "data", "math"),
}

OUTPUT_PATH = os.path.join(ROOT_DIR, "routing", "domain_centroids.pt")


def main():
    print("========================================================================")
    print(" COMPUTING PROTOTYPE DOMAIN CENTROIDS FOR DYNAMIC ROUTING")
    print(" Device: CPU (Zero GPU Contention)")
    print("========================================================================")

    tokenizer, model = get_bert("answerdotai/ModernBERT-Large-Instruct")
    model.to("cpu")
    model.eval()

    tracks_order = ["science", "code", "med", "electrical", "auto", "culinary", "math"]
    centroids_list = []

    for track in tracks_order:
        data_path = DATA_PATHS[track]
        print(f"\nProcessing [{track}] from {data_path}...")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset path missing: {data_path}")

        ds = load_from_disk(data_path)
        split_name = "test" if "test" in ds else "train"
        split_data = ds[split_name]

        # Sample up to 200 questions
        sample_size = min(200, len(split_data))
        questions = [split_data[i]["question"] for i in range(sample_size)]
        print(f"  Encoding {sample_size} validation prompts...")

        # Batch encode
        batch_size = 32
        all_embs = []
        with torch.no_grad():
            for i in range(0, sample_size, batch_size):
                batch_texts = questions[i:i + batch_size]
                inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=256)
                last_hidden = model(**inputs).last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1)
                # Attention-weighted mean pooling
                pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1)
                # L2 normalize each
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                all_embs.append(pooled)

        domain_embs = torch.cat(all_embs, dim=0)  # (N, 1024)
        centroid = domain_embs.mean(dim=0, keepdim=True)  # (1, 1024)
        centroid = torch.nn.functional.normalize(centroid, p=2, dim=1)  # (1, 1024)
        centroids_list.append(centroid)
        print(f"  Domain centroid computed: shape {list(centroid.shape)}, norm={centroid.norm().item():.4f}")

    centroid_matrix = torch.cat(centroids_list, dim=0)  # (6, 1024)

    save_payload = {
        "centroids": centroid_matrix,
        "tracks": tracks_order,
        "tau_route": 0.05,
        "dim": 1024,
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    torch.save(save_payload, OUTPUT_PATH)
    print(f"\nSaved 6-domain centroid tensor matrix to {OUTPUT_PATH}")
    print(f"Shape: {list(centroid_matrix.shape)}")


if __name__ == "__main__":
    main()
