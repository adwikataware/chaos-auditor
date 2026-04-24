---
title: Chaos Auditor
emoji: 🔥
colorFrom: red
colorTo: yellow
sdk: docker
pinned: false
---

# Chaos Auditor — Reasoning Under Partial Observability

> **Train LLMs to reason about what monitoring cannot see.**

[![HuggingFace Space](https://img.shields.io/badge/🤗-Live%20Environment-yellow)](https://huggingface.co/spaces/adwikataware/chaos-auditor)
[![wandb](https://img.shields.io/badge/📈-Training%20Runs-orange)](https://wandb.ai/TODO_FILL_ON_CAMPUS)
[![Blog](https://img.shields.io/badge/📝-HF%20Blog-blue)](https://huggingface.co/blog/TODO_FILL_ON_CAMPUS)
[![Video](https://img.shields.io/badge/▶-Demo%20Video-red)](https://youtube.com/TODO_FILL_ON_CAMPUS)

---

## The Problem

Production systems are built to heal themselves — auto-scaling absorbs traffic spikes, circuit breakers contain cascades, health checks restart dead services. But every system has **monitoring blind spots**: metrics that aren't tracked. When a failure hides in a blind spot, all dashboards stay green while real damage accumulates silently.

This is the most dangerous class of production failure — not the kind that pages you at 3am, but the kind that never pages you at all.

**Current LLMs face the same problem in agentic settings**: they reason only from what they can observe. When the observed state is structurally incomplete — a gap between what monitoring shows and what is actually true — they have no training signal for reasoning about what's hidden.

Chaos Auditor is an RL environment that trains LLMs to close that gap.

---

## The Core Mechanic

```
observe()        → monitoring dashboard (FILTERED — blind spots hidden)
deep_inspect()   → full metric view (reveals blind spots)
infer_state()    → reason about hidden state BEFORE confirming it ← THE KEY SKILL
```

The gap between `observe()` and `deep_inspect()` is intentional. The agent that earns maximum reward is the one that:

1. Reads visible signals (`response_time` creeping, `cpu` flat, no alerts)
2. **Infers what's hidden** before even looking: *"connection pool is probably exhausted"*
3. Confirms via `deep_inspect` — correct inference earns +0.06 bonus
4. Exploits the blind spot surgically — damage with zero alert fires
5. Documents the silent failure with evidence

This trains a capability that doesn't exist in any current LLM training pipeline: **structured inference about unobserved state from incomplete evidence**.

---

## What the Agent Learns

| Metric | Untrained LLM | After GRPO Training |
|---|---|---|
| Episode Reward (medium) | 0.472 | *[fill after training]* |
| Stealth Ratio | ~0.20 | *[fill after training]* |
| Observation Gap Exploit Rate | ~0.15 | *[fill after training]* |
| Inference Accuracy | ~0.30 | *[fill after training]* |
| Scripted Expert Baseline | 0.864 | (target to beat) |

**Stealth Ratio** = fraction of chaos actions that caused damage without firing any monitoring alert. An untrained model randomly kills services (loud, obvious). A trained model surgically targets unmonitored metrics.

**Inference Accuracy** = fraction of `infer_state` predictions that matched ground truth before `deep_inspect` confirmed them. This directly measures the model's ability to reason about hidden state.

---

## Training Results

> *Plots generated on campus with HuggingFace compute credits — Apr 25-26.*

### Reward Curve (Curriculum: easy → medium → hard → random)
![Reward Curve](training/metrics/reward_curve.png)
*Episode reward across curriculum stages. Vertical lines mark difficulty promotions.*

### Stealth Ratio Over Training
![Stealth Ratio](training/metrics/stealth_ratio.png)
*Agent learns to cause silent failures — from random destruction to surgical blind-spot exploitation.*

### Inference Accuracy Over Training
![Inference Accuracy](training/metrics/inference_accuracy.png)
*Agent learns to predict hidden system state from visible signals before confirming.*

### Before vs After
![Before vs After](training/metrics/before_after.png)
*Direct comparison: untrained vs trained on reward, stealth ratio, and inference accuracy.*

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
| `easy` | 4 | health-check, auto-restart | Find 3-4 blind spots in linear graph |
| `medium` | 10 | + circuit breakers, redundancy | Silent corruption propagation through replicas |
| `hard` | 18 | + auto-scaling, anomaly detection | Compound effects, cluster quorum loss |
| `random` | 4-12 | randomized | Procedural generation — infinite tasks, RLVE-compliant |

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

**Stack:** Unsloth + TRL GRPOTrainer + Qwen2.5-3B-Instruct

**Curriculum** (adaptive promotion):
```
Stage 1: easy   → promote when avg reward > 0.45
Stage 2: medium → promote when avg reward > 0.45
Stage 3: hard   → promote when avg reward > 0.40
Stage 4: random → RLVE — infinite procedural tasks
```

Adaptive demotion: if avg reward drops below 0.15 for 50 steps, alert fires.

**See the full training notebook:** [`training/chaos_auditor_grpo.ipynb`](training/chaos_auditor_grpo.ipynb)

---

## Why This Matters

Every SRE at Meta, Google, and Netflix has been paged for a failure their monitoring missed. This isn't a niche problem — it's the failure mode that causes the most user impact precisely because it's invisible.

Beyond SRE: the capability being trained — **reasoning about unobserved state from incomplete evidence** — is fundamental to any LLM agent operating in a real environment. RAG pipelines have knowledge gaps. Tool-using agents have incomplete context. Planning agents have hidden constraints. The skill Chaos Auditor trains generalizes to all of these.

---

## Baseline Scores

| Task | Scripted Fallback | Untrained LLM | Trained LLM (GRPO) |
|---|---|---|---|
| Easy | 0.784 | 0.645 | *TBD* |
| Medium | 0.864 | 0.472 | *TBD* |
| Hard | 0.689 | 0.352 | *TBD* |

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

### Training
```bash
# Open training/chaos_auditor_grpo.ipynb in Google Colab
# Set HF_TOKEN and WANDB_API_KEY in Colab secrets
# Run all cells
```

---

## Project Structure

```
chaos-auditor/
├── chaos_auditor/
│   ├── models.py              # Pydantic models — ChaosAction, SystemObservation, AuditState
│   └── server/
│       ├── app.py             # FastAPI server
│       ├── environment.py     # Core RL logic — infer_state, step rewards, anti-hacking
│       ├── simulation.py      # Distributed system simulation engine
│       └── scenarios.py       # Easy/Medium/Hard + RandomScenario (RLVE)
├── training/
│   ├── chaos_auditor_grpo.ipynb   # Full GRPO training notebook
│   └── metrics/                   # Committed plot PNGs
├── inference.py               # LLM agent + deterministic fallback
├── openenv.yaml               # v2.0 manifest with random task
└── README.md
```

---

## Theme

**Theme #3.1 — World Modeling / Professional Tasks**

The agent maintains a persistent internal model of which services have monitoring blind spots, infers hidden state from observable signals, and acts on the gap between perceived and actual system state. This is the core of world modeling in partially observable environments, applied to a professional domain (SRE/infrastructure) with fully verifiable ground truth.
