import os
import torch
from datasets import load_from_disk
import eval as eval_module
from lora import apply_lora
from system import System2

def extract_cache(split_name, data_dir="./data/arc"):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Extracting {split_name} on device: {device}")
    
    ds = load_from_disk(data_dir)
    split = ds[split_name]
    n = len(split)
    cols = split[:n]
    labels = torch.tensor(cols["label"], device=device)
    
    checkpoint_path = "models/system2.pt"
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    
    model = eval_module.model
    if "lora" in state:
        apply_lora(model, r=32, alpha=32, num_layers=8)
        model.load_state_dict(state["lora"], strict=False)
    
    model.to(device)
    model.eval()
    
    sys2_state = state["sys2"]
    sys2 = System2(
        num_heads=5,
        num_classes=1,
        pairwise=True,
        diff_features=True,
        head_layers=2,
        ensemble_temp=1.78,
        head_weights=[1.0, 1.0, 1.20, 1.0, 0.38],
    )
    sys2.load_state_dict(sys2_state, strict=False)
    sys2.to(device)
    sys2.eval()
    
    batch = 32
    all_logits = []
    all_scores = []
    all_confs = []
    all_embs = []
    
    with torch.no_grad():
        for i in range(0, n, batch):
            b = min(batch, n - i)
            vecs = eval_module.mean_pool_pairwise(
                cols["question"][i : i + b],
                cols["choices"][i : i + b],
                device,
                max_len=128,
                asym_spans=True,
                asym_span_mode="choice_forward",
            )
            out = sys2(vecs)
            best, conf, pruned, confs, logits, emb, scores = out[:7]
            all_logits.append(logits.cpu())
            all_scores.append(scores.cpu())
            all_confs.append(conf.cpu())
            all_embs.append(emb.cpu())
            
    cache_data = {
        "logits": torch.cat(all_logits, dim=0),      # (N, 5, 4)
        "scores": torch.cat(all_scores, dim=0),      # (N, 5)
        "confs": torch.cat(all_confs, dim=0),        # (N,)
        "embs": torch.cat(all_embs, dim=0),          # (N, 5, 512)
        "labels": labels.cpu(),                      # (N,)
    }
    out_path = f"scratch/{split_name}_cache.pt"
    torch.save(cache_data, out_path)
    print(f"Successfully cached {split_name} ({n} samples) to {out_path}")

def main():
    os.makedirs("scratch", exist_ok=True)
    extract_cache("train")
    extract_cache("test")

if __name__ == "__main__":
    main()
