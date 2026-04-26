# Chaos Auditor: Training LLMs to Reason About What Monitoring Can't See

## The Problem

Every production system has monitoring blind spots — metrics that aren't tracked. When a failure hides in a blind spot, all dashboards stay green while real damage accumulates silently. This is the most dangerous class of production failure: not the kind that pages you at 3am, but the kind that never pages you at all.

Current LLMs face the same problem in agentic settings: they reason only from what they can observe. When the observed state is structurally incomplete, they have no training signal for reasoning about what's hidden.

**Chaos Auditor trains LLMs to close that gap.**

## The Environment

Chaos Auditor is an OpenEnv-compliant RL environment simulating a distributed system where:

- `observe()` returns only monitored metrics — the filtered dashboard view
- `deep_inspect(service)` reveals ALL metrics including blind spots
- `infer_state(service, metric, level, reasoning)` lets the agent reason about hidden state **before confirming** — correct inference earns bonus reward

The gap between what monitoring shows and what is actually true is the core training mechanic.

## The Key Innovation: Belief Revision Under Contradiction

Most RL environments reward agents for acting correctly. Chaos Auditor rewards agents for **reasoning correctly before acting** — and for **updating beliefs when evidence demands it**.

The full workflow the agent learns:

1. **State a hypothesis** — *"connection pool is probably exhausted"* — formally, with confidence
2. **Infer before confirming** — predict the hidden metric level before calling `deep_inspect`
3. **Seek disconfirming evidence** — `deep_inspect` may contradict the hypothesis
4. **Revise when contradicted** — +0.03 bonus for correct epistemic update, -0.02 for anchoring
5. **Commit with evidence** — `commit_root_cause` rewards well-evidenced commits, penalizes premature ones
6. **Exploit the blind spot** — silent damage with zero alerts

## Demonstrated Results: Anchoring vs Calibrated Agent

| Metric | Anchoring Agent | Calibrated Agent |
|--------|----------------|-----------------|
| Final score | 0.231 | **0.570** |
| Silent failures found | 0 | **2** |
| Contradictions handled | 0 / 1 | **1 / 1** |
| Stealth ratio | 0.000 | **1.000** |
| Score improvement | — | **+0.339 (+147%)** |

The Calibrated Agent earns higher reward not because it knew the answer — but because it **updated its belief when evidence contradicted its hypothesis**. This is exactly the capability Chaos Auditor trains.

## Training Results

### Reward Curve (Real Training Run)
![Reward Curve](training/metrics/reward_curve.png)

### Before vs After GRPO Training
![Before vs After](training/metrics/before_after.png)

| Metric | Untrained | Trained | Change |
|--------|-----------|---------|--------|
| Episode Reward | 0.005 | 0.012 | **+140%** |

## Training Setup

- **Model**: Qwen2.5-1.5B-Instruct
- **Algorithm**: GRPO (Group Relative Policy Optimization)
- **Curriculum**: easy (4 services) → medium (10 services) → hard (18 services) → random (RLVE)
- **SFT Warmup**: 4 demonstration trajectories teach action format before RL starts
- **Tracking**: [Weights & Biases Run](https://wandb.ai/sohamtakale2905-mit-world-peace-university/chaos-auditor-grpo?nw=nwusersohamtakale2905)

## Why It Matters

The capability being trained — reasoning about unobserved state from incomplete evidence — is fundamental to any LLM agent operating in a real environment:

- **RAG pipelines** have knowledge gaps
- **Tool-using agents** have incomplete context
- **Planning agents** have hidden constraints

Chaos Auditor trains the skill that makes agents reliable in all of these settings.

## Links

- **Environment + Demo**: [HuggingFace Space](https://huggingface.co/spaces/adwikataware/chaos-auditor)
- **Training Notebook**: [Google Colab](https://colab.research.google.com/github/adwikataware/chaos-auditor/blob/main/training/chaos_auditor_grpo.ipynb)
- **Training Run**: [Weights & Biases](https://wandb.ai/sohamtakale2905-mit-world-peace-university/chaos-auditor-grpo?nw=nwusersohamtakale2905)
- **Code Repository**: [GitHub](https://github.com/adwikataware/chaos-auditor)
