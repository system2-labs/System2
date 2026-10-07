import os
import torch
from datasets import load_from_disk
import eval as eval_module
from lora import apply_lora
from system import System2

def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    ds = load_from_disk("./data/arc")
    test = ds["test"]
    n = len(test)
    cols = test[:n]
    labels = torch.tensor(cols["label"], device=device)
    
    checkpoint_path = "models/system2.pt"
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    
    model = eval_module.model
    if "lora" in state:
        cfg = state.get("lora_config", {"r": 32, "alpha": 32, "num_layers": 8})
        train_norms = cfg.get("train_norms", False) or ("lora_norms" in state)
        apply_lora(model, r=cfg.get("r", 32), alpha=cfg.get("alpha", 32), num_layers=cfg.get("num_layers", 8), train_norms=train_norms)
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
        ensemble_temp=1.80,
        head_weights=[1.0, 1.0, 1.25, 1.0, 0.35],
    )
    sys2.load_state_dict(sys2_state, strict=False)
    sys2.to(device)
    sys2.eval()
    
    batch = 32
    head_correct = torch.zeros(5, dtype=torch.long, device=device)
    any_correct = torch.zeros(n, dtype=torch.bool, device=device)
    
    all_logits = []
    all_scores = []
    all_confs = []
    all_ens_logits = []
    
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
            best, conf, pruned, confs, logits, emb, scores, ens_logits = out
            preds_per_head = logits.argmax(dim=-1) # (B, 5)
            batch_labels = labels[i : i + b].unsqueeze(-1) # (B, 1)
            correct_matrix = (preds_per_head == batch_labels) # (B, 5)
            
            head_correct += correct_matrix.sum(dim=0)
            any_correct[i : i + b] = correct_matrix.any(dim=1)
            
            all_logits.append(logits.cpu())
            all_scores.append(scores.cpu())
            all_confs.append(conf.cpu())
            all_ens_logits.append(ens_logits.cpu())
            
    print(f"\nTotal test samples: {n}")
    for h in range(5):
        acc = head_correct[h].item() / n
        print(f"Head {h} accuracy: {acc*100:.2f}% ({head_correct[h].item()}/{n})")
    
    oracle_acc = any_correct.sum().item() / n
    print(f"\nORACLE HEAD ACCURACY (at least one head correct): {oracle_acc*100:.2f}% ({any_correct.sum().item()}/{n})")
    
    cat_ens_logits = torch.cat(all_ens_logits, dim=0)
    ens_preds = cat_ens_logits.argmax(dim=-1)
    ens_acc = (ens_preds == labels.cpu()).float().mean().item()
    print(f"Current System2 Ensemble Accuracy: {ens_acc*100:.2f}% ({(ens_preds == labels.cpu()).sum().item()}/{n})")
    
    os.makedirs("scratch", exist_ok=True)
    cache_data = {
        "logits": torch.cat(all_logits, dim=0),      # (3530, 5, 4)
        "scores": torch.cat(all_scores, dim=0),      # (3530, 5)
        "confs": torch.cat(all_confs, dim=0),        # (3530,)
        "labels": labels.cpu(),                      # (3530,)
    }
    torch.save(cache_data, "scratch/test_cache.pt")
    print("Cached logits and scores to scratch/test_cache.pt")

if __name__ == "__main__":
    main()
