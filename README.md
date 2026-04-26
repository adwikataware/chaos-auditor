---
title: Chaos Auditor
emoji: 🔥
colorFrom: red
colorTo: yellow
sdk: docker
pinned: false
---

# Chaos Auditor — Belief Revision Under Partial Observability

> **Train LLMs to form hypotheses, seek disconfirming evidence, and revise beliefs when the evidence demands it.**

[![HuggingFace Space](https://img.shields.io/badge/🤗-Live%20Environment-yellow)](https://huggingface.co/spaces/adwikataware/chaos-auditor)
[![Colab](https://img.shields.io/badge/📓-Training%20Notebook-orange)](https://colab.research.google.com/drive/1D0EcWRnQTNysf0n6rK7KClyRHFicFI1O?usp=sharing)
[![Blog](https://img.shields.io/badge/📝-Blog%20Post-blue)](https://huggingface.co/spaces/adwikataware/chaos-auditor/blob/main/Blog.md)
[![wandb](https://img.shields.io/badge/📈-Training%20Run-yellow)](https://wandb.ai/sohamtakale2905-mit-world-peace-university/chaos-auditor-grpo?nw=nwusersohamtakale2905)
[![GitHub](https://img.shields.io/badge/💻-GitHub-black)](https://github.com/adwikataware/chaos-auditor)

---

## The Problem

Production failures that monitoring misses are invisible by design. Finding them requires forming hypotheses, actively seeking contradicting evidence, and revising beliefs when the evidence demands it. Most LLMs can't do this — they anchor on first impressions and ignore contradiction.

This is the same failure mode seen in deployed LLM agents: they commit to their first hypothesis and filter evidence through it, rather than updating when new information arrives. **Belief revision under contradiction** is a capability no current training pipeline explicitly targets.

**Chaos Auditor is an RL environment that trains this capability explicitly** — in a domain with fully verifiable ground truth, measurable before/after results, and reward signals that penalize anchoring and reward genuine belief revision.

The agent maintains a persistent world model of which services have monitoring blind spots, forms hypotheses about hidden state from visible signals, and updates those beliefs as evidence arrives. This is world modeling in a partially observable professional environment — the capability generalizes beyond SRE to any LLM agent operating with incomplete context.

---

## The Core Mechanic

```
observe()                            → monitoring dashboard (FILTERED — blind spots hidden)
deep_inspect()                       → full metric view (reveals blind spots)
infer_state()                        → predict hidden state BEFORE confirming ← inference skill
state_hypothesis()                   → formally commit to a root cause belief ← belief tracking
revise_hypothesis()                  → update belief after contradicting evidence ← THE KEY SKILL
commit_root_cause()                  → commit with evidence trail
```

The gap between `observe()` and `deep_inspect()` is intentional. The agent that earns maximum reward is the one that:

1. Reads visible signals (`response_time` creeping, `cpu` flat, no alerts)
2. **States a hypothesis**: *"connection pool is probably exhausted"* — formally, with confidence
3. **Seeks disconfirming evidence** via `deep_inspect` — not confirming evidence
4. **Revises when contradicted**: environment flags contradictions, +0.03 for correct revision
5. **Commits with evidence** when confidence is sufficient — penalized for premature commits
6. Exploits the blind spot surgically — damage with zero alert fires

This trains two capabilities that don't exist in any current LLM training pipeline:
- **Structured inference about unobserved state** from incomplete evidence
- **Belief revision under contradiction** — updating beliefs when evidence demands it

---

## What the Agent Learns

| Metric | Random Agent | Scripted Fallback | After GRPO Training |
|---|---|---|---|
| Episode Reward (easy) | 0.042 | 0.580 | **0.012** |
| Episode Reward (medium) | 0.001 | 0.437 | **0.010** |
| Stealth Ratio | ~0.52 | 1.000 | **0.50** |
| Observation Gap Exploit Rate | ~0.07 | 1.000 | **0.80** |
| Inference Accuracy | 0.000 | 0.000 | **0.12** |
| Hypothesis Revision Rate | 0.000 | 0.000 | **0.40** |

*GRPO training: Qwen2.5-1.5B-Instruct, 40 curriculum updates, easy→medium→hard→random. Full run tracked at [Weights & Biases](https://wandb.ai/sohamtakale2905-mit-world-peace-university/chaos-auditor-grpo).*

*Baselines measured on 10 held-out seeds per task. Scripted fallback uses no LLM — hardcoded action sequences targeting known blind spots.*

**Stealth Ratio** = fraction of chaos actions that caused damage without firing any monitoring alert. An untrained model randomly kills services (loud, obvious). A trained model surgically targets unmonitored metrics.

**Inference Accuracy** = fraction of `infer_state` predictions that matched ground truth before `deep_inspect` confirmed them. Directly measures the model's ability to reason about hidden state.

**Hypothesis Revision Rate** = fraction of contradiction events where the agent correctly revised its belief instead of anchoring. Directly measures the belief revision capability — the key metric.

---

## Training Results

> *Plots generated on campus with HuggingFace compute credits — Apr 25-26.*

### Reward Curve (Curriculum: easy → medium → hard → random)
![Reward Curve](training/metrics/reward_curve.png)
*Real training run. Episode reward across curriculum stages. Vertical lines mark difficulty promotions.*

### Before vs After GRPO Training
![Before vs After](training/metrics/before_after.png)
*Untrained: 0.005 avg reward. After GRPO curriculum: 0.012 (+140%). Measured on held-out seeds.*

### Agent Comparison: Anchoring vs Calibrated
![Agent Comparison](training/metrics/agent_comparison.png)
*Scripted demo showing what the trained behavior looks like. Calibrated agent earns +147% higher score purely from belief revision.*

---

## Environment Design

### The Partial Observability Gap

Every service has two views:

```python
get_monitoring_view()   # Only tracked metrics — what dashboards show
get_deep_view()         # All metrics — including blind spots
```

Blind spots are per-service. A database might not monitor `connection_count`. A cache might not monitor `data_integrity`. The agent must discover these gaps and exploit them.

### Action Space

**Chaos Actions** (cost 1 budget each)
| Action | Effect | Silent if... |
|---|---|---|
| `kill` | Service goes DOWN | Never (status alert fires) |
| `spike_traffic` | CPU/connections spike | Scaling absorbs it |
| `corrupt_data` | Data integrity degrades | `data_integrity` not monitored |
| `add_latency` | Response time increases | Below alert threshold |
| `partition_network` | Communication blocked | Circuit breaker masks it |
| `fill_disk` | Disk fills up | `disk_usage` not monitored |
| `exhaust_connections` | Connection pool fills | `connection_count` not monitored |

**Free Actions** (no budget cost)
| Action | Purpose |
|---|---|
| `observe` | View monitoring dashboard (filtered) |
| `deep_inspect(service)` | Reveal ALL metrics including blind spots |
| `infer_state(service, metric, level, reasoning)` | **Reason about hidden state before confirming** |
| `state_hypothesis(root_cause, confidence, reasoning)` | Formally commit to a belief |
| `revise_hypothesis(root_cause, new_confidence, reason)` | Update belief after contradiction — **+0.03 reward** |
| `commit_root_cause(root_cause, evidence_summary)` | Commit with evidence trail |
| `classify_finding` | Document a vulnerability |
| `submit_report` | End episode, trigger final scoring |

### Reward Design

**Step-level shaping** (intermediate signals at every step):
| Signal | Reward |
|---|---|
| `deep_inspect` reveals new blind spot | +0.02 |
| Chaos action targets a known blind spot | +0.03 |
| Chaos causes damage, zero alerts fire | +0.05 |
| `infer_state` correct (blind metric) | +0.06 |
| `infer_state` correct (monitored metric) | +0.02 |
| `infer_state` wrong | −0.02 |
| `revise_hypothesis` after contradiction | +0.03 |
| `commit_root_cause` with evidence | +0.02 |
| Premature commit (low confidence) | −0.02 to −0.03 |
| Redundant `deep_inspect` (same service) | −0.01 |
| Chaos on fully-monitored service (after step 5) | −0.02 |

**Episode-level scoring**:
| Signal | Reward |
|---|---|
| Finding matches ground truth vulnerability | up to 0.35 per finding |
| Efficiency bonus (≤50% budget used) | +0.08 |
| Stealth bonus (zero total alerts) | +0.08 |
| Inference mastery (≥60% accuracy, ≥2 attempts) | +0.05 |
| False finding penalty | −0.05 each |

**Reward hacking prevention**:
- Same finding type submitted >2 times → −0.05 penalty
- `classify_finding` for services never acted on or inspected → −0.03 (coherence gate)
- `infer_state` after already inspecting the service → −0.01

### Tasks

| Task | Services | Defenses | Key Challenge |
|---|---|---|---|
| `easy` | 4 | health-check, auto-restart | Find 4 blind spots in linear graph |
| `medium` | 10 | + circuit breakers, redundancy | Silent corruption propagation through replicas |
| `hard` | 18 | + auto-scaling, anomaly detection | Compound effects, cluster quorum loss |
| `random` | 4–12 | randomized | Procedural generation — infinite tasks, RLVE-compliant |

### Simulation Features

- **Service dependency graph** with cascading failure propagation
- **Connection pool drain** — connections leak under sustained stress
- **Memory leaks** — gradual memory growth under stress over multiple ticks
- **Request queue backpressure** — slow downstream causes upstream queuing
- **Circuit breakers** — CLOSED → OPEN → HALF-OPEN state machine
- **Compound effects** — two sub-threshold attacks create emergent silent failures
- **Data corruption propagation** — corrupted cache spreads to all dependent services
- **Procedural scenario generator** — randomized topology, blind spots, defenses (RLVE)

---

## Training Pipeline

**Stack:** PEFT + manual GRPO loop + Qwen2.5-1.5B-Instruct + QLoRA (4-bit)

**Two-phase training**:
```
Phase 1 — SFT warmup (2 epochs)
  Generate ~250 demonstrations from scripted fallback agent (4 plan variants)
  Teach action format and belief-revision workflow before GRPO starts

Phase 2 — GRPO curriculum
  Stage 1: easy   → promote when avg reward > 0.45
  Stage 2: medium → promote when avg reward > 0.45
  Stage 3: hard   → promote when avg reward > 0.40
  Stage 4: random → RLVE — infinite procedural tasks
```

GRPO reward function runs **full multi-step episodes** — not single actions. This is critical: belief-revision actions (`state_hypothesis`, `revise_hypothesis`) return 0.0 reward on step 1 but enable +0.03–+0.08 downstream. Single-step scoring would teach the model to ignore them.

Adaptive demotion: if avg reward drops below 0.15 for a stage, alert fires.

**See the full training notebook:** [`training/chaos_auditor_grpo.ipynb`](training/chaos_auditor_grpo.ipynb)

---

## Why This Matters

Every SRE at Meta, Google, and Netflix has been paged for a failure their monitoring missed. This isn't a niche problem — it's the failure mode that causes the most user impact precisely because it's invisible.

Beyond SRE: the capability being trained — **reasoning about unobserved state from incomplete evidence** — is fundamental to any LLM agent operating in a real environment. RAG pipelines have knowledge gaps. Tool-using agents have incomplete context. Planning agents have hidden constraints. The skill Chaos Auditor trains generalizes to all of these.

---

## Baseline Scores

*Measured on 10 held-out seeds per task using `eval_harness.py`.*

| Task | Random Agent | Scripted Fallback | Trained LLM (GRPO) |
|---|---|---|---|
| Easy | 0.042 | 0.580 | **0.012** (+140% vs untrained 0.005) |
| Medium | 0.001 | 0.437 | **0.010** |
| Hard | — | 0.249 | **0.007** |

---

## Setup

### Docker (recommended)
```bash
docker build -t chaos-auditor .
docker run -p 8000:8000 chaos-auditor
```

### Local Development
```bash
pip install -e .
uvicorn chaos_auditor.server.app:app --host 0.0.0.0 --port 8000
```

### Pre-flight (run before training)
```bash
python preflight.py   # 18 checks — validates env, all actions, contradiction detection
```

### Eval Harness
```bash
python eval_harness.py --mode scripted        # scripted fallback baseline
python eval_harness.py --mode random_agent    # random action baseline
python eval_harness.py --mode llm --model-path ./chaos-auditor-trained
```

### Training
```bash
# Open training/chaos_auditor_grpo.ipynb in Google Colab
# Set HF_TOKEN and WANDB_API_KEY in Colab secrets
# Run all cells (Cell 4b = SFT warmup, Cell 10+ = GRPO curriculum)
```

---

## Project Structure

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
│   ├── chaos_auditor_grpo.ipynb   # Full SFT + GRPO training notebook
│   └── metrics/                   # Committed plot PNGs
├── inference.py               # LLM agent + deterministic fallback
├── play_demo.py               # Before/after trajectory comparison (no GPU needed)
├── preflight.py               # Pre-training validation — run before compute
├── eval_harness.py            # Reproducible eval with held-out seeds
├── openenv.yaml               # v2.0 manifest
└── README.md
```

---

## Theme

**Theme #3.1 — World Modeling / Professional Tasks**

The agent maintains a persistent internal model of which services have monitoring blind spots, infers hidden state from observable signals, and acts on the gap between perceived and actual system state. This is the core of world modeling in partially observable environments, applied to a professional domain (SRE/infrastructure) with fully verifiable ground truth.
