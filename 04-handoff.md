# System2 - Handoff Document

Read this first. It is the current state of the project, what works, what is
broken, and exactly what to do next. Written 2026-09-24.

---

## 1. What this project is

A from-scratch "System 2" model: parallel hypotheses, calibrated confidence,
pruning of weak ones. The only pre-trained component is ModernBERT-large, used
**purely as an encoder** (text -> 1024-dim vector). It has no generation
ability, no RLHF layer, no safety overlay. Everything downstream (heads,
judger) is untrained and gets trained here, on this Mac.

Why from scratch: building on an autoregressive model inherits its alignment
layer, biases, and confident-wrong personality. That is the exact failure
mode Selsam's Sep 2026 letter warns about.

Full design context (read later, not needed for the next steps):
- 01-system-architecture.md - the two paths (build vs augment), architecture
- 02-training-plan.md - training pipelines, datasets, milestones
- 03-implementation-spec.md - detailed code-level spec (partially superseded
  by what is actually on disk now - trust the .py files over the spec)

## 2. What is on disk (current state)

    ~/system2/
    bert.py        encoder: ModernBERT-large, FROZEN. encode() -> (1, 1024)
    heads.py       ThoughtHead: 1024->512, 2 transformer layers,
                   confidence head (BCE-calibrated) + hypothesis head
    system.py      System2: N ThoughtHeads + Judger (heads attend to each
                   other; best score wins; weak ones pruned at <0.3).
                   forward() returns 7-tuple: (best, best_conf, pruned,
                   confs, logits, embeddings, scores)  [B,N dims where batched]
    data.py        AG News loader (from ./data/ag_news on disk)
    eval.py        THE JUDGE (read-only): accuracy + Brier (mislabelled ECE in
                   v1, TODO bins), metric = 0.5*acc + 0.5*(1-Brier), JSON line
    train.py       THE SUBJECT: supervised v1 (CE + confidence-BCE + judger
                   loss). HYPERPARAMS block + CLI overrides are what the loop
                   tunes. 20,000 samples, 2 epochs, device mps.
                   SHUFFLES data before slicing (splits are sorted).
    train_smoke.py THIN WRAPPER around train.py (200 samples, 1 epoch canary)
    runner.sh      AutoResearch loop: train -> judge -> commit better /
                   git reset worse. (from test-training repo pattern)
    data/ag_news/  AG News on disk (train + test splits, HF format)
    01/02/03-*.md  planning docs

Verified working (run today):
- bert.py: encode() -> torch.Size([1, 1024])  OK
- heads.py: token 15362, confidence 0.479 (random untrained - expected)  OK
- system.py: 5 heads, picks 1, reports confs  OK

Verified working (run 2026-09-24 on M4 Pro):
- train_smoke.py: 1 epoch / 200 samples, 24s on MPS, saves models/system2.pt  OK
- eval.py: loads checkpoint, recovers num_heads from state, prints JSON metric  OK
- MPS is ~15x faster than CPU for the encoder (32ms/text vs 490ms/text at b=64)

NOT done: git init (runner.sh needs it), first real train.py run, .best_metric,
models/ dir.

## 3. Do this, in this order

### Step 0 - git init (runner.sh depends on it)

    cd ~/system2
    git init
    # .gitignore FIRST, so the big dataset never gets staged:
    echo "data/ag_news/" >> .gitignore   # ~100MB, lives in the HF cache anyway
    echo "__pycache__/" >> .gitignore
    echo "*.pyc" >> .gitignore
    # NOTE: do NOT gitignore .best_metric - it is the loop's memory, track it.
    git add -A
    git commit -m "feat(system2): v1 scaffold - encoder + heads + judger + AG News pipeline"

    # optional: create a remote on GitHub and git remote add origin ... + git push -u origin main

### Step 1 - smoke test (canary, 200 samples)

    python3 train_smoke.py

Expect: ~30-60s, one epoch line (ce/cal/judge losses), saves ./models/system2.pt.
Losses after 200 samples will look noisy (ce ~1.4-3.7 is normal at this size).
If it errors, read the traceback - do NOT hand-edit around it; the bug is real.
train_smoke.py is now a THIN WRAPPER around train.py - change logic in train.py only.

### Step 2 - record baseline

    ./runner.sh "baseline"

This trains (20k samples, 2 epochs, MPS ~= 12 min), runs eval on all 7600 test
rows, prints:
    [runner] new metric = 0.XXXX  ({"metric": ..., "accuracy": ..., "brier": ...})
First run records .best_metric and commits.

Known reference point (UNTRAINED heads, 500 test samples): acc 0.254.
Sanity targets after 2 epochs on 20k (calibrated expectation, not promise):
    accuracy 0.60-0.75 on AG News with this architecture
    metric 0.55-0.70
If accuracy < 0.45 after the full run, something is wrong - check train.py
loss trajectory (ce should fall below ~0.5 within epoch 2 on this task).

### Step 3 - the AutoResearch loop (the actual project)

Now the loop from test-training, applied to hyperparameters:

    1. Edit the HYPERPARAMS block in train.py (ONLY that block):
       num_heads, lr_heads, lr_judger, batch_size, epochs, train_size
    2. Run ./runner.sh "describe your change"
    3. It commits if the metric improved, reverts if not.

Starting hypotheses worth testing (in rough order of expected value):
    - lr_heads: 5e-4, 2e-3
    - num_heads: 3, 8
    - batch_size: 16, 64
    - epochs: 1, 3
    - max_len: 64, 256 (AG News headlines are short; 128 may already be enough)

Do NOT edit: bert.py, eval.py, system.py, heads.py architecture, data/.
That is the rule from program.md in test-training: the judge is untouchable,
or the metric is meaningless. If you suspect a bug in the judge, fix it and
note it - the baseline metric gets invalid and .best_metric must be deleted
so the loop re-baselines.

## 4. Known issues

1. (Fixed this session) eval.py and train_smoke.py tokenizer calls were
   missing padding=True - batched encoding raised ValueError. Patched.
   train.py had padding=True already.

2. (Environment quirk, not yours) the assistant output pipeline corrupts the
   token sequence "arg" + "max" into *** in some code paths - a long debugging
   session was wasted on this. If a line in a .py file shows up as ***, do not
   fight the assistant on it; fix it locally:
       python3 -c "
       import pathlib; p = pathlib.Path('FILE'); L = p.read_text().split(chr(10))
       L[LINE-1] = '        hypo_token=logits.a' + 'rgmax(dim=-1)'
       p.write_text(chr(10).join(L))"
   (or just sed it). This is a rendering bug, not a file problem.

3. (Open design question) v1 trains confidence with BCE against 1[head is
   correct]. This is a supervised stand-in for Laya-style RLCD. Fine for v1.
   If calibration (ECE) plateaus bad, swap in REINFORCE per 03-implementation-spec
   section 3.2.

4. (Fixed 2026-09-24) AG News HF splits are SORTED BY LABEL. train[:N] is a
   single class. train.py now does ds["train"].shuffle(seed=0) BEFORE slicing.
   If you touch data loading, keep the shuffle - or the metric becomes meaningless.

5. (Pitfall) __pycache__ can hold a stale system.py and Python silently uses it.
   If behaviour does not match the source, run:  rm -rf __pycache__

6. (Fixed) eval.py recovers num_heads from the saved state, so the loop can
   vary num_heads in train.py without the judge breaking.

7. (Open) num_heads=5 fixed at construction. Dynamic head count per query
   (simple queries need 3, hard ones need 10) is on the roadmap in
   01-system-architecture.md section 10.3. Not next.

## 5. After AG News works

The milestone that matters: metric clearly above a single-ModernBERT-CLF
baseline (fine-tune ModernBERT-large + linear head on AG News for comparison -
one evening of work). If System2 with 5 heads + judger beats or matches a
single linear probe while reporting calibrated confidence, the architecture
is validated and we move to reasoning (GSM8K) per 02-training-plan.md.

If it does NOT beat the linear probe: the parallel-hypothesis machinery is
dead weight for this task. That is a useful result - it tells us pruning
helps for harder tasks. Do not keep tuning it into the baseline on an easy
task.

## 6. Invariants (do not break these)

- Encoder stays frozen. Full-model training of 395M params on a Mac is the
  "a week" path in 02-training-plan.md and not needed yet.
- eval.py stays a read-only oracle. The moment the loop can edit the judge,
  the whole experiment is invalid (same rule as prepare.py in test-training).
- No autoregressive component. If you add text generation, that is a new
  design decision, not an implementation detail.
- One file = one concept. bert/heads/system are importable modules with
  __main__ demos. Keep it that way - it is what makes each piece testable.

## 7. Where the big ideas live (for later sessions)

- The Selsam letter argument (eval-awareness) motivated all of this:
  ~.hermes/composer-pastes/pasted_content_2026-09-24_06-21-12-081_b3d18f.txt
- Laya (the model this is spiritually closest to):
  https://huggingface.co/convaiinnovations/laya
- Karpathy autoresearch pattern (source of the loop):
  https://github.com/tj-doomer-it/test-training
- Apple runtimes if we ever need inference speed:
  https://github.com/incoai/splash  https://github.com/youssofal/MTPLX
