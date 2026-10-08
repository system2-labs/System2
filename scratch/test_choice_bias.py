import torch
from datasets import load_from_disk
import eval as eval_module
from lora import apply_lora
from system import System2

def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Testing on device: {device}")
    
    ds = load_from_disk("./data/arc")
    test = ds["test"]
    n = len(test)
    cols = test[:n]
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
    
    # 1. Normal evaluation (Question + Answer)
    # We already have test_cache.pt for normal logits!
    normal_data = torch.load("scratch/test_cache.pt", weights_only=True)
    normal_logits = normal_data["logits"] # (3530, 5, 4)
    normal_scores = normal_data["scores"]
    hw = torch.tensor([1.0, 1.0, 1.20, 1.0, 0.38])
    w = torch.softmax(normal_scores / 1.78 + torch.log(hw), dim=-1)
    base_ens = torch.einsum('bn,bnc->bc', w, normal_logits)
    base_preds = base_ens.argmax(dim=-1)
    base_acc = (base_preds == labels.cpu()).float().mean().item()
    print(f"Normal Question+Answer Accuracy: {base_acc*100:.2f}% ({(base_preds == labels.cpu()).sum().item()}/{n})")
    
    # 2. Choice-Only Evaluation: What does the model predict if given an empty or dummy question?
    # e.g. Question: "" Answer: choices
    blank_questions = ["" for _ in range(n)]
    batch = 32
    all_blank_logits = []
    
    with torch.no_grad():
        for i in range(0, n, batch):
            b = min(batch, n - i)
            vecs = eval_module.mean_pool_pairwise(
                blank_questions[i : i + b],
                cols["choices"][i : i + b],
                device,
                max_len=128,
                asym_spans=True,
                asym_span_mode="choice_forward",
            )
            out = sys2(vecs)
            logits = out[4] # (B, 5, 4)
            all_blank_logits.append(logits.cpu())
            
    blank_logits = torch.cat(all_blank_logits, dim=0) # (3530, 5, 4)
    blank_ens = torch.einsum('bn,bnc->bc', w, blank_logits)
    blank_preds = blank_ens.argmax(dim=-1)
    blank_acc = (blank_preds == labels.cpu()).float().mean().item()
    print(f"Choice-Only (Blank Question) Accuracy: {blank_acc*100:.2f}% ({(blank_preds == labels.cpu()).sum().item()}/{n}) (Random chance is 25.0%)")
    
    # 3. Debiased Evaluation: z_debiased = normal_logits - beta * blank_logits
    print("\n--- Testing Choice-Only Bias Subtraction ---")
    for beta in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 1.0]:
        debiased_logits = normal_logits - beta * blank_logits
        debiased_ens = torch.einsum('bn,bnc->bc', w, debiased_logits)
        d_preds = debiased_ens.argmax(dim=-1)
        d_acc = (d_preds == labels.cpu()).float().mean().item()
        d_brier = ((normal_data["confs"] - (d_preds == labels.cpu()).float()) ** 2).mean().item()
        d_metric = 0.5 * d_acc + 0.5 * (1 - d_brier)
        print(f"beta={beta:.2f} -> Acc={d_acc*100:.2f}% ({(d_preds == labels.cpu()).sum().item()}/{n}), Brier={d_brier:.4f}, Metric={d_metric:.4f}")

if __name__ == "__main__":
    main()
