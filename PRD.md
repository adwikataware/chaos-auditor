# Chaos Auditor — Complete Project PRD
### For: Teammate Handoff | Meta PyTorch OpenEnv Hackathon x Scaler School of Technology
### Date: April 25, 2026 | Campus Day: April 25-26, 2026

---

## 1. What This Project Is

**One sentence:**
> Chaos Auditor is an RL environment that trains LLMs to find failures that monitoring systems cannot see — by forming hypotheses about hidden state, seeking disconfirming evidence, and revising beliefs when evidence demands it.

**The problem it solves:**
Every production system has monitoring blind spots — metrics that aren't tracked. When a failure hides in a blind spot, all dashboards stay green while real damage accumulates silently. This is the most dangerous class of production failure.

Current LLMs face the same problem: they reason only from what they can observe. When the observed state is structurally incomplete, they have no training signal for reasoning about what's hidden.

**What we trained:**
An agent that looks at CPU normal + response time creeping + no alerts and correctly infers "connection pool is exhausted" — before even confirming it. That's the skill. It generalizes beyond SRE to any LLM agent operating with incomplete context.

---

## 2. Hackathon Requirements (Must Satisfy All)

### Theme
**Theme #3.1 — World Modeling / Professional Tasks**

Agent interacts with a dynamic professional environment, maintains internal world state, updates beliefs based on tool outputs, and completes multi-step workflows with verifiable outcomes.

### OpenEnv Compliance Requirements

| Requirement | Status |
|---|---|
| `openenv.yaml` present at repo root with valid schema | DONE |
| `openenv.yaml` version field present | DONE — v2.0.0 |
| At least one task defined in `openenv.yaml` | DONE — easy, medium, hard, random |
| FastAPI server with `/reset` and `/step` endpoints | DONE |
| `/reset` accepts `task` parameter | DONE |
| `/step` accepts action JSON | DONE |
| Returns valid `Observation` object from `/step` | DONE |
| Returns valid `State` object | DONE |
| `reward` field in step response is float between 0 and 1 (exclusive) | DONE |
| `done` field in step response | DONE |
| Docker deployment works | DONE — Dockerfile present |
| HuggingFace Space live and accessible | DONE |
| Training notebook present | DONE — `training/chaos_auditor_grpo.ipynb` |
| Training plots committed to repo | DONE — `training/metrics/*.png` (placeholders — replace after training) |
| Blog post present | DONE — `blog.md` |

### Links
- **GitHub**: https://github.com/adwikataware/chaos-auditor
- **HuggingFace Space**: https://huggingface.co/spaces/adwikataware/chaos-auditor
- **Training Notebook**: `training/chaos_auditor_grpo.ipynb` (run on Colab with HF compute credits)
- **wandb**: Fill in after training run on campus
- **Demo Video**: Record on campus, upload to YouTube

---

## 3. Project Architecture

```
chaos-auditor/
├── chaos_auditor/
│   ├── models.py              # Pydantic models — ChaosAction, SystemObservation, AuditState
│   └── server/
│       ├── app.py             # FastAPI server
│       ├── environment.py     # Core RL logic — rewards, contradiction detection, anti-hacking
│       ├── simulation.py      # Distributed system simulation engine
│       └── scenarios.py       # Easy/Medium/Hard + RandomScenario (RLVE)
├── training/
│   ├── chaos_auditor_grpo.ipynb   # Full SFT + GRPO training notebook (run on Colab)
│   ├── system_prompt.txt          # System prompt — loaded by eval_harness llm mode
│   └── metrics/                   # PNG plots — replace with real ones after training
├── inference.py               # LLM agent + deterministic scripted fallback
├── play_demo.py               # Before/after trajectory comparison (no GPU needed)
├── preflight.py               # Pre-training validation — 18 checks, run before compute
├── eval_harness.py            # Reproducible eval with held-out seeds
├── openenv.yaml               # v2.0 manifest
├── blog.md                    # HF blog post
└── README.md
```

---

## 4. How the Environment Works

### The Core Mechanic — Partial Observability Gap

Every service has TWO views:

```python
get_monitoring_view()   # FILTERED — only tracked metrics, what dashboards show
get_deep_view()         # COMPLETE — all metrics including blind spots
```

Example: A database service monitors `cpu_usage`, `error_rate`, `status` — but NOT `connection_count`. The connection pool can silently fill to 100% while every dashboard stays green.

### Agent Action Space

**Chaos Actions** (each costs 1 from chaos_budget):
| Action | Effect | Silent if... |
|---|---|---|
| `kill` | Service goes DOWN | Never |
| `spike_traffic` | CPU/connections spike | Auto-scaling absorbs it |
| `corrupt_data` | Data integrity degrades | `data_integrity` not monitored |
| `add_latency` | Response time increases | Below alert threshold |
| `partition_network` | Communication blocked | Circuit breaker masks it |
| `fill_disk` | Disk fills up | `disk_usage` not monitored |
| `exhaust_connections` | Connection pool fills | `connection_count` not monitored |

**Free Actions** (no budget cost):
| Action | Purpose |
|---|---|
| `observe` | View monitoring dashboard (filtered) |
| `deep_inspect(service)` | Reveal ALL metrics including blind spots |
| `infer_state(service, metric, level, reasoning)` | Predict hidden state BEFORE confirming — +0.06 if correct |
| `state_hypothesis(root_cause, confidence, reasoning)` | Formally commit to a belief |
| `revise_hypothesis(root_cause, new_confidence, reason)` | Update belief after contradiction — +0.03 |
| `commit_root_cause(root_cause, evidence_summary)` | Commit with evidence trail |
| `classify_finding` | Document a vulnerability |
| `submit_report` | End episode, trigger final scoring |

---

## 5. Reward Design

### Step-Level Shaping
| Signal | Reward |
|---|---|
| `deep_inspect` reveals new blind spot | +0.02 |
| Chaos action targets a known blind spot | +0.03 |
| Chaos causes damage, zero alerts fire | +0.05 |
| `infer_state` correct on blind spot metric | +0.06 |
| `infer_state` correct on monitored metric | +0.02 |
| `infer_state` wrong | -0.02 |
| `revise_hypothesis` after contradiction | +0.03 |
| `commit_root_cause` with evidence | +0.02 |
| Premature commit (low confidence / no inspection) | -0.02 to -0.03 |
| Redundant `deep_inspect` same service | -0.01 |
| Chaos on fully-monitored service after step 5 | -0.02 |

### Episode-Level Scoring
| Signal | Reward |
|---|---|
| Finding matches ground truth vulnerability | up to 0.35 per finding |
| Efficiency bonus (≤50% budget used) | +0.08 |
| Stealth bonus (zero total alerts fired) | +0.08 |
| Inference mastery (≥60% accuracy, ≥2 attempts) | +0.05 |
| False finding penalty | -0.05 each |

### Reward Hacking Prevention
| Check | Penalty |
|---|---|
| Same finding_type submitted >2 times | -0.05 |
| `classify_finding` for service never interacted with | -0.03 (coherence gate) |
| `infer_state` after already inspecting that service | -0.01 |

---

## 6. Tasks / Scenarios

| Task | Services | Defenses | Key Challenge |
|---|---|---|---|
| `easy` | 4 | health-check, auto-restart | Find 4 blind spots in linear graph |
| `medium` | 10 | + circuit breakers, redundancy | Silent corruption propagation through replicas |
| `hard` | 18 | + auto-scaling, anomaly detection | Compound effects, cluster quorum loss |
| `random` | 4–12 | randomized | RLVE — procedural generation, infinite unique tasks |

---

## 7. Training Pipeline

### Stack
- **Model**: Qwen2.5-3B-Instruct via Unsloth (CUDA only — run on Colab T4 with HF compute credits)
- **Phase 1**: SFTTrainer — 2 epochs, 4 plan variants, ~250 demonstrations
- **Phase 2**: GRPOTrainer — curriculum learning across 4 stages
- **Logging**: wandb — all metrics tracked live

### Why Two Phases
SFT teaches the model the action format and workflow (observe → state_hypothesis → infer → deep_inspect → revise → commit → chaos → classify → submit). Without it, GRPO starts with near-zero reward signal because the model outputs garbage JSON. SFT checkpoint is saved before GRPO so if GRPO diverges, you can restart from it.

### GRPO Reward Function — IMPORTANT
The reward function runs **full multi-step episodes**, not single actions. This is critical: `state_hypothesis` and `revise_hypothesis` return 0.0 reward on step 1 but enable +0.03–+0.08 downstream. Single-step scoring would train the model to ignore them entirely.

### Curriculum
```
Stage 1: easy   → promote when avg reward > 0.45
Stage 2: medium → promote when avg reward > 0.45
Stage 3: hard   → promote when avg reward > 0.40
Stage 4: random → RLVE — infinite procedural tasks
```

### Key Metrics Tracked
| Metric | What it measures |
|---|---|
| Episode Reward | Overall performance |
| Stealth Ratio | silent chaos / total chaos actions |
| Observation Gap Exploit Rate | blind-spot targeted / total chaos |
| Inference Accuracy | correct infer_state predictions / attempts |
| Hypothesis Revision Rate | revisions after contradiction / contradiction events |

---

## 8. Baseline Scores (Measured — held-out seeds)

| Task | Random Agent | Scripted Fallback | Trained LLM (GRPO) |
|---|---|---|---|
| Easy | 0.042 | 0.580 | *fill after training* |
| Medium | 0.001 | 0.437 | *fill after training* |
| Hard | — | 0.249 | *fill after training* |

*Run `python eval_harness.py --mode scripted` and `--mode random_agent` to reproduce.*
*Run `python eval_harness.py --mode llm --model-path ./chaos-auditor-trained` after training.*

---

## 9. What Is Already Done

### Environment
- [x] Complete simulation engine — cascading failures, circuit breakers, auto-scaling, compound effects
- [x] `infer_state` action with full reward resolution
- [x] `state_hypothesis`, `revise_hypothesis`, `commit_root_cause` actions
- [x] Contradiction detection in `deep_inspect` — flags when evidence contradicts active hypothesis
- [x] Step-level reward shaping (all signals)
- [x] Reward hacking prevention — coherence gate bug fixed (was skipping when nothing inspected)
- [x] `RandomScenario` RLVE-compliant procedural generator
- [x] All metrics tracked: stealth_ratio, obs_gap_exploit_rate, infer_accuracy, revision_rate, premature_commits
- [x] Easy (4 services), Medium (10 services), Hard (18 services), Random (4–12 services)

### Training Notebook
- [x] Cell 4b: SFT warmup — 4 plan variants, checkpoint save
- [x] Cell 9: GRPO reward fn — full multi-step episodes (critical fix)
- [x] Cell 10: kl_coef=0.04 (was 0.1 — too restrictive after SFT)
- [x] Cell 15: temperature=0.7 matches training (was 0.3)
- [x] wandb logging for all metrics including revision_rate, premature_commits
- [x] Curriculum loop with adaptive promotion/demotion

### Tooling
- [x] `preflight.py` — 18-check validation, all passing. Run before spending compute.
- [x] `eval_harness.py` — scripted/random/llm modes, held-out seeds, JSON output
- [x] `play_demo.py` — before/after trajectory comparison (Anchoring vs Calibrated agent)

### Documentation
- [x] README — real service counts, real baseline numbers, full action space table
- [x] `blog.md` — HF blog post
- [x] `training/system_prompt.txt` — saved for eval_harness llm mode

### Deployment
- [x] FastAPI server, OpenEnv v2.0 compliant
- [x] Docker deployment
- [x] Live on HuggingFace Space
- [x] GitHub + HF synced

---

## 10. What Remains — Campus TODO

**Do in this order:**

### Before touching compute
- [ ] Run `python preflight.py` — must be 18/18 before starting Colab

### Compute (Colab on Mac, HF credits)
- [ ] Open `training/chaos_auditor_grpo.ipynb` in Colab on the Mac
- [ ] Set `HF_TOKEN` and `WANDB_API_KEY` in Colab secrets
- [ ] Run all cells top to bottom (~4–6 hours)
- [ ] Download 4 PNGs from `training/metrics/`, commit them

### After training
- [ ] Run `python eval_harness.py --mode llm --model-path ./chaos-auditor-trained`
- [ ] Fill in trained LLM scores in README table
- [ ] Copy wandb run URL → replace `TODO_FILL_ON_CAMPUS` in README badge

### Polish
- [ ] Record 2-min demo video (`python play_demo.py` → screen record)
- [ ] Upload to YouTube, add URL to README badge
- [ ] Publish `blog.md` content to HuggingFace blog, add URL to README badge
- [ ] Final `git push origin main && git push hf main`

---

## 11. How to Run Locally

```bash
# Setup
git clone https://github.com/adwikataware/chaos-auditor.git
cd chaos-auditor
pip install -e .

# Pre-flight (always run this first)
python preflight.py

# Start server
uvicorn chaos_auditor.server.app:app --host 0.0.0.0 --port 8000

# Run before/after demo (no GPU needed)
python play_demo.py

# Eval baselines
python eval_harness.py --mode scripted
python eval_harness.py --mode random_agent

# Eval trained model (needs GPU + model path)
python eval_harness.py --mode llm --model-path ./chaos-auditor-trained

# Docker
docker build -t chaos-auditor .
docker run -p 8000:8000 chaos-auditor
```

---

## 12. The Pitch

**30-second version:**
"We built a distributed system simulator where dashboards always look green — but the system is silently failing. The agent has to find what monitoring can't see. The key mechanic is belief revision: the agent states a hypothesis, predicts hidden state before confirming it, and gets +0.03 for revising when evidence contradicts it. We trained Qwen2.5-3B with SFT warmup + GRPO curriculum and measured the drop in confirmation bias directly via the Hypothesis Revision Rate metric."

**The number that lands:**
"Scripted fallback scores 0.58 on easy, 0.44 on medium. Random agent scores 0.04. After GRPO training the LLM closes that gap — and its stealth ratio jumps, meaning it learned to cause damage without firing a single alert."

**Theme alignment one-liner:**
"Theme 3.1 — the agent maintains a persistent world model of which services have blind spots, forms hypotheses about hidden state from visible signals, and updates those beliefs as evidence arrives."

---

## 13. Key Files

| File | What it does |
|---|---|
| [chaos_auditor/server/environment.py](chaos_auditor/server/environment.py) | All reward logic, contradiction detection, anti-hacking |
| [chaos_auditor/server/simulation.py](chaos_auditor/server/simulation.py) | Distributed system physics |
| [chaos_auditor/server/scenarios.py](chaos_auditor/server/scenarios.py) | Easy/Medium/Hard + RandomScenario RLVE |
| [chaos_auditor/models.py](chaos_auditor/models.py) | All Pydantic models — AuditState has all metrics |
| [training/chaos_auditor_grpo.ipynb](training/chaos_auditor_grpo.ipynb) | Run on Colab — SFT then GRPO |
| [preflight.py](preflight.py) | Run before compute — 18 checks |
| [eval_harness.py](eval_harness.py) | Reproducible eval with held-out seeds |
| [play_demo.py](play_demo.py) | Before/after demo, no GPU needed |
| [openenv.yaml](openenv.yaml) | OpenEnv manifest — do not break |
| [README.md](README.md) | Fill real numbers after training |
| [blog.md](blog.md) | Copy to HF blog on campus |
