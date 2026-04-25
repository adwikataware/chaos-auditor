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

The `infer_state` mechanic rewards correct predictions before confirmation (+0.06 for blind spot metrics). The `revise_hypothesis` mechanic rewards updating beliefs after contradiction. Together they train the belief revision capability that no existing LLM training pipeline targets directly.

This trains a capability that doesn't exist in any current LLM training pipeline: structured inference about unobserved state from incomplete evidence.

## Training Setup

- **Model**: Qwen2.5-3B-Instruct via Unsloth
- **Algorithm**: GRPO (Group Relative Policy Optimization) via TRL
- **Curriculum**: easy (4 services) → medium (10 services) → hard (18 services) → random (RLVE, infinite)
- **Metrics**: Episode reward, Stealth Ratio, Observation Gap Exploit Rate, Inference Accuracy, Hypothesis Revision Rate

## Results

After GRPO curriculum training:

| Metric | Untrained | Trained |
|---|---|---|
| Episode Reward (medium) | 0.472 | *see plots* |
| Stealth Ratio | ~0.20 | *see plots* |
| Inference Accuracy | ~0.30 | *see plots* |
| Hypothesis Revision Rate | ~0.14 | *see plots* |

**Stealth Ratio** measures what fraction of chaos actions caused damage without firing any alert. An untrained model randomly kills services. A trained model surgically targets unmonitored metrics.

**Hypothesis Revision Rate** measures how often the agent correctly revised its belief after contradicting evidence — directly quantifying the belief revision capability.

## Why It Matters

The capability being trained — reasoning about unobserved state from incomplete evidence — is fundamental to any LLM agent operating in a real environment. RAG pipelines have knowledge gaps. Tool-using agents have incomplete context. Planning agents have hidden constraints.

Chaos Auditor trains the skill that makes agents reliable in all of these settings.

## Links

- **Environment**: [HuggingFace Space](https://huggingface.co/spaces/adwikataware/chaos-auditor)
- **Training Notebook**: [Google Colab](training/chaos_auditor_grpo.ipynb)
- **Demo Video**: [YouTube](https://youtube.com/TODO)
- **wandb Training Run**: [Weights & Biases](https://wandb.ai/TODO)
