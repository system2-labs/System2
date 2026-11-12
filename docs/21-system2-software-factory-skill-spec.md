# System 2 Sovereign Software Factory & 10-Pass Lifecycle Specification

## 1. Overview & Paradigm

The **System 2 Software Factory** is an on-device, autonomous development lifecycle designed to eliminate continuous human monitoring and reduce cloud data center energy consumption.

### The Developer Workflow:
1. **Grounded Planning**: Shape the project plan interactively via a frontier model OR the local `big-brain-hybrid` profile.
2. **Plan Packaging**: Package the plan into the standardized 10-pass executive roles and deterministic milestone criteria.
3. **Hardware Ejection & Profile Handoff**:
   - **Workstations with 128GB–512GB RAM**: `big-brain-hybrid` stays memory-resident in VRAM as an active supervisor/reviewer alongside subagents.
   - **Systems with <=128GB RAM (or 24GB-48GB VRAM)**: `big-brain-hybrid` completes the plan packaging, unloads/ejects itself to free RAM/VRAM, then triggers `fast-coding` (24GB) or `cybersec-redteam` (24GB) to execute passes locally.
4. **Step Away / Offline Local Build**: The human developer disconnects, drinks coffee, does laundry, and lets the local hardware build autonomously.
5. **Adversarial Triad Review**: Gatekeeper (AST check), Heretic (adversarial fuzzing), and Sentinel (security & immutability invariants) evaluate each pass with sub-85ms automatic rollback on regression.
6. **Scorecard Verification & Next Iteration**: Review the empirical scorecard report, verify deliverables, and initiate the next cycle.

---

## 2. Hardware Memory Tiering & Profile Lifecycle

| Profile | Target VRAM | Primary Cognitive Role | Backbones Loaded | Memory Strategy |
| :--- | :--- | :--- | :--- | :--- |
| **`big-brain-hybrid`** | 128 GB | Deep Architectural Planning & Specification | `qwen3.8-27b-splash` (local) + `qwen3.8-flash-next` (remote/heavy) + `lfm2.5-1.2b-instruct` | **High RAM (128–512GB)**: Keep resident in memory.<br>**Constrained (<=128GB)**: Eject after planning phase. |
| **`fast-coding`** | 24 GB | Rapid TDD, Code Generation & AST Refactoring | `qwen-3.8-27b-intermediate@iq3_xxs` + `gemma-4-e2b-it-qat` + `lfm2.5-1.2b-instruct` | Standard daily driver; runs concurrently with adversarial triad. |
| **`cybersec-redteam`** | 24 GB | Security Auditing, SQLi Fuzzing & Invariant Proofs | `deepseek-v4-pro-qwen3.5-9b-uncensored-heretic` + `Huihui-Qwen3.8-27B-abliterated` + `lfm3-2b-omni` | Deployed for Pass 6 (Security), pen-testing, and compliance gating. |
| **`lean-battery`** | 10 GB | Mobile / Battery-Constrained Execution | `qwen3.5-9b` + `gemma-4-e2b-it-qat` + `lfm2.5-1.2b-instruct` | Ultralight edge profile for untethered laptop operation. |

---

## 3. The 10-Pass Lifecycle & Scorecard Template

```markdown
### 10-Pass Software Factory Lifecycle & Scorecard

Pass │ Role                 │ Milestone                                          │ Latency │ Status │ Software Factory Triad Verdict
─────┼──────────────────────┼────────────────────────────────────────────────────┼─────────┼────────┼────────────────────────────────────────────────────
  01 │ CTO / Database       │ Relational Schema & Ingestion                      │  4.3ms  │ ✓ PASS │ GREEN LIGHT: Zero NULL primary keys, age bounded in
     │ Architect            │ Normalize 206 contacts into SQLite 3NF schema with │         │        │ [18, 120], tables created with compound indices.
     │                      │ foreign keys and compound indices on (city, occ).  │         │        │
  02 │ Full-Stack Lead      │ FastAPI Core REST Engine                           │ 200.8ms │ ✓ PASS │ GREEN LIGHT: Pydantic validation, parameterized
     │                      │ Constructed typed ASGI API (app.py) serving        │         │        │ queries, and 404 entity handling.
     │                      │ /api/health, /api/users, /api/users/{id}.          │         │        │
  03 │ Lead Quant / Data    │ Multivariate Contact Matching                      │  1.2ms  │ ✓ PASS │ GREEN LIGHT: Proved similarity score S(u, v) ∈ [0, 1]
     │ Scientist            │ Implemented matching_engine.py with city affinity, │         │        │ monotonically decreasing with age delta.
     │                      │ occupation alignment, and Gaussian decay.          │         │        │
  04 │ Product Lead         │ Sovereign Social Web UI                            │  0.1ms  │ ✓ PASS │ GREEN LIGHT: Zero external CDN dependencies for
     │                      │ Responsive CSS Grid single-page app (index.html)   │         │        │ 100% on-device sovereign operation.
     │                      │ with Profile, Feed, and Suggestions components.    │         │        │
  05 │ Founder / Growth     │ EdgeRank Feed Optimization                         │  1.2ms  │ ✓ PASS │ GREEN LIGHT: Validated strict descending score
     │ Architect            │ EdgeRank scoring (U × W × D) with affinity, like   │         │        │ order across synthesized researcher posts.
     │                      │ weights, and temporal decay.                       │         │        │
  06 │ Security Red Team    │ Fail-Closed Security & Invariants                  │  0.2ms  │ ✓ PASS │ GREEN LIGHT: Blocked SQLi attempts (' OR '1'='1),
     │                      │ SQL injection probes, demographic range checks,    │         │        │ 0 out-of-bound ages, 0 raw PII credentials.
     │                      │ and PII protection tests.                          │         │        │
  07 │ Head of Finance /    │ Unit Economics & Sovereign Edge TCO                │  0.6ms  │ ✓ PASS │ GREEN LIGHT: Proved 94.7% operational cost reduction
     │ CFO                  │ 3-year TCO comparing cloud frontier API spend vs.  │         │        │ ($9,270 cloud vs $540 edge) and positive margins.
     │                      │ local edge execution.                              │         │        │
  08 │ Systems Engineer     │ Async Event Bus Dispatch Integration               │  1.7ms  │ ✓ PASS │ GREEN LIGHT: Verified async handler execution and
     │                      │ Integrated test/event_bus.py with typed            │         │        │ error isolation semantics.
     │                      │ UserConnectedEvent dispatch.                       │         │        │
  09 │ QA Lead              │ Pytest End-to-End Test Suite                       │ 732.0ms │ ✓ PASS │ GREEN LIGHT: 4/4 integration modules passing
     │                      │ Automated integration tests (test_social_suite.py) │         │        │ with exit code 0.
     │                      │ covering DB integrity, API routes, and matching.   │         │        │
  10 │ System 2 Specialist  │ Cognitive Scorecard & Audit                        │  0.0ms  │ ✓ PASS │ GREEN LIGHT: 10/10 passes verified under
     │                      │ Verified all 4 System 2 components, measured KPIs, │         │        │ adversarial criteria; audited cross-repo diffs.
     │                      │ and audited cross-repository discrepancies.        │         │        │
```

---

## 4. Discrepancy Tracking Matrix Template

Whenever agents audit cross-repository or cross-subsystem code during execution, document discrepancies in this standardized table:

```markdown
### Discrepancies Identified & Fixed Across Subsystems

Subsystem             │ Discrepancy Found                         │ Impact                                   │ Resolution Implemented
──────────────────────┼───────────────────────────────────────────┼──────────────────────────────────────────┼──────────────────────────────────────────
Daemon Bridge Port    │ Hardcoded port mismatch (e.g. 8765 vs     │ Client reports local daemon offline      │ Dual-port candidate fallback: checks
                      │ 8000).                                    │ during initial health check.             │ primary port, auto-falls back in 60ms.
Sys1 Lexical Patterns │ Restrictive keyword filtering matching    │ Natural SWE queries default to 'general' │ Lexical expansion: added domain terms
                      │ only raw code fences.                     │ track instead of 'code'.                 │ (repo, script, csv, test, api, sqlite).
Cognitive Elevation   │ Deep deliberation only on explicit manual │ Multi-step refactors bypass 5-Head       │ Dynamic elevation: complex tasks auto-
                      │ virtual model selection.                  │ deliberation pipeline.                   │ elevate to Sys2 5-Head arbitration.
TUI Screen Layout     │ Dynamic footer line counts or raw strings │ Soft line-wrapping causing scrollback    │ Strict 3-Line Layout: fixed line height,
                      │ overflowing terminal width.               │ duplication cascades.                    │ hard clipping via truncateToWidth.
```

---

## 5. Subagent Dispatch & Execution Guidelines

When orchestrating System 2 subagents:
1. **Always invoke with explicit persona headers**:
   - `[Arbiter // Sandbox]` for bash commands and environment tests.
   - `[Philosopher // Refactor]` for file edits and code generation.
   - `[Wanderer // Search]` for file inspection, grep, and codebase exploration.
2. **Command Dispatch Format**:
   ```bash
   system2 --provider system2 --model auto -p "As the <Role>, <Actionable Task Description with Acceptance Criteria>"
   ```
3. **Adversarial Triad Review Command**:
   Run the 3-agent review triad after each pass before advancing:
   - **Gatekeeper**: AST syntax and type correctness (`python3 -m py_compile <file>`).
   - **Heretic**: Boundary tests and adversarial inputs.
   - **Sentinel**: AST invariant verification and security checks.
4. **Rollback Law**:
   If any pass fails the acceptance criteria or breaks existing tests, trigger atomic worktree rollback immediately (`git checkout -- <file>` or `system2 rollback`). Do not commit regressions.
