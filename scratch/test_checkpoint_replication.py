import torch
import subprocess
import json

state = torch.load("models/system2.pt", map_location="cpu", weights_only=True)
state["ensemble_temp"] = 1.78
state["head_weights"] = [1.0, 1.0, 1.20, 1.0, 0.38]
torch.save(state, "scratch/test_cand.pt")
print("Saved scratch/test_cand.pt with ensemble_temp=1.78, head_weights=[1.0, 1.0, 1.20, 1.0, 0.38]")

out = subprocess.check_output(["python3", "eval.py", "scratch/test_cand.pt"], text=True)
print("Eval output:", out.strip())
