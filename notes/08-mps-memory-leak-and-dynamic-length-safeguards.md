# Note 08: MPS Memory Accumulation, Dynamic Sequence Lengths, and Memory Safeguards

## 1. Incident Summary
During an AutoResearch training run on the legal reasoning dataset (`runner_legal.sh`), the Python training process (PID 35896) consumed **159.2 GB** of physical RAM (peak **164.7 GB**) and **664 GB** of virtual address space. 

This exhausted system unified memory (driving 93 GB of wired memory and 15 GB of compressed memory) and triggered severe disk swap thrashing (>19M swapouts). The Metal GPU stream stalled in `-[_MTLCommandBuffer waitUntilCompleted]` waiting on OS paging during tensor scalar synchronization (`.item()`).

---

## 2. Root Cause Analysis

### A. CLI Parameter Overwrite Bug (`--max-len` vs `--dynamic-len`)
In `train.py`, the CLI argument parsing contained an inadvertent overwrite:
```python
# Before fix:
if args.max_len is not None:
    HYPERPARAMS["max_len"] = args.max_len
if args.dynamic_len:
    HYPERPARAMS["dynamic_len"] = True
    HYPERPARAMS["max_len"] = 8192  # Overwrote any explicit --max-len (e.g. 1024)!
```
Because `--dynamic-len` defaulted to `True`, passing `--max-len 1024` was completely discarded, allowing token sequences to scale up to the full 8192 ModernBERT context ceiling on legal texts.

### B. PyTorch Metal (MPS) Caching Allocator Fragmentation
When training with dynamic sequence lengths (`padding=True, truncation=True`), each batch possesses a different maximum sequence length $T$ (e.g., 210, 485, 890, 1024). 
- PyTorch's Metal Performance Shaders (MPS) allocator creates `MTLBuffer` allocations sized to the current batch dimensions.
- Because shapes vary across batches, the allocator cannot reuse existing cached blocks of different sizes.
- Unlike CUDA, the default MPS allocator on macOS does not aggressively garbage-collect or release unused cached heap blocks back to the OS during the training loop.
- Over hundreds of batches across multiple epochs, cached Metal allocations accumulate continuously until memory is exhausted.

### C. Autograd Graph & Gradient Retention
- `optimizer.zero_grad()` by default zeroes gradient tensors rather than deallocating them (`set_to_none=True`).
- Intermediate local variables (`total_loss`, `vecs`, `out`) remained referenced in the Python loop frame across the start of subsequent iterations, extending the autograd backward graph lifetime across step boundaries.

---

## 3. Engineering Fixes Applied

### 1. Correct Argument Hierarchy
`args.max_len` now strictly overrides `HYPERPARAMS["max_len"]` regardless of whether `--dynamic-len` is enabled:
```python
if args.dynamic_len is not None:
    HYPERPARAMS["dynamic_len"] = args.dynamic_len
if args.max_len is not None:
    HYPERPARAMS["max_len"] = args.max_len
elif args.dynamic_len and "max_len" not in HYPERPARAMS:
    HYPERPARAMS["max_len"] = 8192
```

### 2. MPS Process Memory Ceiling
At training initialization, an upper bound fraction is placed on MPS allocations:
```python
if is_mps and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    if hasattr(torch.mps, "set_per_process_memory_fraction"):
        try:
            torch.mps.set_per_process_memory_fraction(0.7)
        except Exception:
            pass
```

### 3. Gradient Deallocation & Active Tensor Cleanup
- Switched to `optimizer.zero_grad(set_to_none=True)` to deallocate gradient memory immediately.
- Explicitly delete batch tensors (`total_loss`, `task_loss`, `vecs`, `lab`, `out`) before the next iteration.

### 4. Periodic Metal Allocator Cache Purge
Every 10 batches and at the boundary of each epoch, the MPS allocator cache is flushed:
```python
if is_mps and n_batches % 10 == 0:
    torch.mps.empty_cache()

# At epoch completion:
if is_mps:
    torch.mps.empty_cache()
```

### 5. OS-Level Environment Variable Guardrail
Added to runner scripts (`runner.sh`, `runner_legal.sh`):
```bash
export PYTORCH_MPS_HIGH_WATERMARK_RATIO="${PYTORCH_MPS_HIGH_WATERMARK_RATIO:-0.7}"
```
This instructs the Metal driver to trim its buffer pool whenever allocations exceed 70% of available system memory.
