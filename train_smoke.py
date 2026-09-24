"""train_smoke.py - CANARY: same pipeline as train.py, tiny data.

Thin wrapper only: import train.main and override the size knobs.
If you change training logic, change it in train.py - keep exactly one
copy of the logic (lesson from 2026-09-24: two copies drift).
"""
import sys

import train

if __name__ == "__main__":
    # Default to smoke settings, then let CLI args override.
    sys.argv = [a for a in sys.argv if a != "--smoke"]
    if "--train-size" not in sys.argv:
        sys.argv += ["--train-size", "200"]
    if "--epochs" not in sys.argv:
        sys.argv += ["--epochs", "1"]
    train.main()
