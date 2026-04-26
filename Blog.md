# Chaos Auditor: Teaching AI to Find What Monitoring Misses

In 2017, Netflix's chaos engineering team ran an experiment. They injected a failure into their system and watched. The dashboards stayed green. No alerts fired. But real users were experiencing degraded video quality for 40 minutes before anyone noticed.

The failure was real. The monitoring just couldn't see it.

This is the problem we built Chaos Auditor to solve.

---

## The Problem

Production systems have blind spots. Metrics that aren't tracked. Thresholds that aren't set. Failures that cause real damage while every dashboard stays green.

Current LLMs make this worse. Ask one to diagnose a system and it locks onto its first hypothesis, then filters every new piece of evidence through that lens. It doesn't look for proof that it's wrong. It looks for proof that it's right.

That's anchoring bias. And in a partially observable system, it gets you killed.

---

## What We Built

Chaos Auditor is an RL environment that trains an LLM to do the opposite.

The agent gets two views of a distributed system:

observe() gives the monitoring dashboard. Filtered. What ops teams actually see.

deep_inspect(service) gives everything. Including the metrics nobody is watching.

The gap between those two calls is where silent failures live. The agent's job is to find that gap, reason about what's hiding in it, and exploit it without firing a single alert.

---

## The Workflow the Agent Learns

1. Read the dashboard and form a hypothesis with a confidence level
2. Predict what a hidden metric looks like before confirming it
3. Run deep_inspect and check if reality contradicts the hypothesis
4. If it does, revise. Don't anchor. Update the belief and the confidence.
5. Commit to a root cause only when the evidence is solid
6. Hit the blind spot surgically. Silent damage. Zero alerts.

Step 4 is the hard one. It's the capability no current training pipeline targets explicitly.

---

## The Reward Signal

Every reward is tied to reasoning quality, not just outcomes.

Correct prediction about a hidden metric before confirming: +0.06

Silent chaos action (damage caused, no alert fired): +0.05

Hitting a known blind spot: +0.03

Revising hypothesis after contradiction: +0.03

Committing root cause with solid evidence: +0.02

Wrong prediction: -0.02

Premature commit before evidence: -0.02

The agent can't hack this. Killing a monitored service fires an alert and gets penalized. The only path to high reward is the right path: reason carefully, update beliefs, act silently.

---

## Training

Model: Qwen2.5-1.5B-Instruct with 4-bit QLoRA

Two phases:

Phase 1 is SFT warmup. Four demonstration trajectories showing the full belief revision workflow. This gives the model enough signal that GRPO has something to work with.

Phase 2 is GRPO curriculum. Easy (4 services) to medium (10 services) to hard (18 services) to random procedural tasks. Each stage only promotes when the model has actually learned the previous one.

Untrained baseline: 0.005 average reward

After GRPO curriculum: 0.012 (+140%)

The scripted fallback agent (no LLM, hardcoded logic) scores 0.58. That gap is what the training is closing over more episodes.

---

## Anchoring vs Calibrated: The Demo

Same environment. Same seed. Two agents.

The anchoring agent locks on network partition as its hypothesis. Inspects the database, finds contradiction, ignores it. Kills a monitored service anyway. Alert fires. Score: 0.231.

The calibrated agent states connection pool exhaustion as its hypothesis. Inspects the database. Finds that connection count is actually fine but disk usage is high and unmonitored. Revises. Commits with evidence. Fills the disk silently. No alert. Score: 0.570.

That is +147% from one capability: updating a belief when evidence says you are wrong.

---

## Why This Generalizes

SRE is the domain. The capability is universal.

Any LLM agent operating with incomplete context faces this problem. RAG pipelines have knowledge gaps. Tool-using agents have missing tool results. Planning agents have hidden constraints.

The skill Chaos Auditor trains is reasoning correctly about what you cannot see, and changing your mind when new information arrives. That works everywhere.

---

## Links

Environment and live demo: [HuggingFace Space](https://huggingface.co/spaces/adwikataware/chaos-auditor)

Training notebook: [Google Colab](https://colab.research.google.com/github/adwikataware/chaos-auditor/blob/main/training/chaos_auditor_grpo.ipynb)

Training metrics: [Weights and Biases](https://wandb.ai/sohamtakale2905-mit-world-peace-university/chaos-auditor-grpo)

Code: [GitHub](https://github.com/adwikataware/chaos-auditor)
