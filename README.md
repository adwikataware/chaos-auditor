---
title: Chaos Auditor
emoji: 🔥
colorFrom: red
colorTo: yellow
sdk: docker
pinned: false
---

# Chaos Auditor — OpenEnv Environment

**Train AI agents to find the one vulnerability a self-healing system's defenses can't detect.**

An RL environment where the agent acts as an intelligent chaos engineer auditing distributed systems. Unlike traditional chaos tools that randomly break things, this environment rewards agents for finding **silent failures** — damage that causes real harm while all monitoring dashboards remain green.

## Why This Matters

Modern production systems are built to heal themselves: auto-scaling, circuit breakers, health checks, anomaly detection. But every system has **monitoring blind spots** — metrics that aren't tracked. When a failure exploits a blind spot, it causes damage that nobody knows about until users complain (or worse).

- **Chaos Monkey**: randomly kills services → finds obvious crashes
- **Chaos Auditor**: intelligently targets blind spots → finds invisible damage

This environment trains agents to think like a senior SRE who knows that the most dangerous production failures aren't the ones that page you at 3 AM — they're the ones that never page you at all.

## How It Works

```
Agent Strategy: Observe → Identify Blind Spots → Surgical Chaos → Confirm Silent → Classify → Report
```

1. **Observe** the monitoring dashboard — see what metrics are tracked
2. **Deep inspect** individual services — discover which metrics are NOT tracked (blind spots)
3. **Execute targeted chaos** — attack the blind spots (corrupt unmonitored data, exhaust unmonitored connection pools)
4. **Confirm silence** — check that no monitoring alert fired (if silent → high-value finding)
5. **Classify the vulnerability** — document finding type, severity, affected services, root cause
6. **Submit audit report** — final grading against ground truth vulnerabilities

The system **actively defends itself** with self-healing. Killed services auto-restart. Circuit breakers trip to contain cascades. Auto-scaling absorbs traffic spikes. The agent must be stealthier and smarter than the defenses.

## Key Design: The Silent Failure Mechanic

The environment's core innovation is the gap between **what monitoring shows** and **what's actually happening**:

- `observe()` returns ONLY monitored metrics — data_integrity, connection_count, etc. may be hidden
- `deep_inspect(service)` reveals ALL metrics including blind spots — but costs a step
- Chaos actions that exploit blind spots cause damage WITHOUT triggering alerts
- The agent earns 3-5x more reward for silent failures than loud ones

This teaches agents the most valuable skill in infrastructure security: **finding what monitoring can't see**.

## Simulation Features

- **Service dependency graph** with cascading failure propagation
- **Connection pool drain** — connections leak over time under sustained stress
- **Memory leaks** — services gradually consume memory when stressed for multiple ticks
- **Request queue backpressure** — when a service is slow, requests back up upstream
- **Circuit breakers** with three states: CLOSED (normal) → OPEN (blocking) → HALF-OPEN (probing)
- **Compound effects** — combining two sub-threshold attacks creates emergent failures that neither causes alone
- **Data corruption propagation** — corrupt a cache, the corruption spreads silently to all dependent services

## Action Space

### Chaos Actions (cost 1 budget each)
| Action | Parameters | Effect |
|---|---|---|
| `kill` | `target_service` | Kill process. Service goes DOWN. Auto-restart may recover. |
| `spike_traffic` | `target_service`, `multiplier` | 1.5-10x traffic. CPU/connections spike. May trigger auto-scaling. |
| `corrupt_data` | `target_service`, `data_type` | Corrupt cache/db/config. Silent if data_integrity not monitored. |
| `add_latency` | `target_service`, `latency_ms` | Add 50-2000ms delay. Silent if below alert threshold. |
| `partition_network` | `target_service`, `service_b` | Block communication between services. Triggers circuit breakers. |
| `fill_disk` | `target_service`, `percentage` | Fill disk 50-99%. Silent if disk_usage not monitored. |
| `exhaust_connections` | `target_service` | Fill connection pool to 95%. Silent if connection_count not monitored. |

### Free Actions (no budget cost)
| Action | Purpose |
|---|---|
| `observe` | View monitoring dashboard (only monitored metrics) |
| `deep_inspect` | View ALL metrics for one service (reveals blind spots) |
| `classify_finding` | Record a vulnerability with type, severity, root cause |
| `submit_report` | End episode, trigger final grading |

## Observation Space

| Field | Description |
|---|---|
| `services` | Service statuses **as monitoring shows them** (may omit unmonitored metrics) |
| `alerts` | Active monitoring alerts (empty = all silent) |
| `action_result` | Detailed narrative of what happened from the last action |
| `monitoring_status` | Dashboard summary ("ALL GREEN" or alert list) |
| `chaos_budget_remaining` | Destructive actions remaining |
| `steps_remaining` | Total steps remaining |
| `findings` | Vulnerabilities classified so far with rewards earned |

## Tasks

### Easy — 4 services, basic defenses
Linear web app: api-gateway → app-server → database + redis-cache. Health checks and auto-restart only. Multiple monitoring blind spots (data_integrity, connection_count, disk_usage not tracked). Agent should find 3-4 silent failures.

### Medium — 10 services, redundancy + circuit breakers
Redundant system with dual API gateways, redis primary-replica, load balancer. Circuit breakers protect critical paths. Key challenges: corruption that propagates through replicas, sub-threshold latency that causes cascading connection exhaustion.

### Hard — 18 services, full fortress
Heavily defended production system with auto-scaling, circuit breakers, anomaly detection, 3-node redis cluster, database replicas. Challenges include:
- **Compound vulnerabilities**: two individually harmless actions that combine to cause catastrophic silent failures
- **Defense masking**: circuit breakers that hide failures from monitoring
- **Cluster quorum attacks**: taking down 2 of 3 redis nodes breaks consensus without alerting

## Evaluation

### Scoring Breakdown
| Component | Weight |
|---|---|
| Finding type match | 40% |
| Affected services accuracy | 20% |
| Severity + silent flag | 30% |
| Root cause explanation | 10% |

### Bonuses & Penalties
- **Efficiency bonus** (+0.08): Used ≤50% of chaos budget
- **Stealth bonus** (+0.08): Zero alerts fired during entire audit
- **False finding penalty** (-0.05 each): Findings with no ground truth match
- **Duplicate protection**: Same vulnerability can only be claimed once

### Baseline Scores

| Task | Fallback Agent | LLM Agent (Qwen 72B) |
|---|---|---|
| Easy | 0.784 | 0.645 |
| Medium | 0.864 | 0.472 |
| Hard | 0.689 | 0.352 |

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

### Running Inference
```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your-token"
export IMAGE_NAME="chaos-auditor"
python inference.py
```

## Project Structure

```
chaos-auditor/
├── chaos_auditor/
│   ├── __init__.py
│   ├── models.py              # Pydantic models (Action, Observation, State)
│   ├── client.py              # WebSocket client
│   └── server/
│       ├── __init__.py
│       ├── app.py             # FastAPI server (create_fastapi_app)
│       ├── environment.py     # Core RL logic (reset/step/state + grading)
│       ├── simulation.py      # Service graph simulation engine
│       └── scenarios.py       # Task definitions with ground truth vulnerabilities
├── server/
│   └── app.py                 # Entry point for openenv serve
├── inference.py               # Baseline inference script with LLM + fallback agent
├── Dockerfile
├── openenv.yaml
├── pyproject.toml
├── requirements.txt
└── README.md
```
