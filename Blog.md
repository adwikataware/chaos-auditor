# Chaos Auditor: Teaching AI to Find What Monitoring Misses

In 2017, Netflix's chaos engineering team ran an experiment. They injected a failure into their system and watched. The dashboards stayed green. No alerts fired. But real users were experiencing degraded video quality for 40 minutes before anyone noticed.

The failure was real. The monitoring just couldn't see it.

This is the problem we built Chaos Auditor to solve. We're a team of engineers who got genuinely obsessed with this question during the Meta OpenEnv Hackathon: can you train an LLM to reason about what it cannot see? Not just react to visible signals, but form beliefs, seek disconfirming evidence, and update those beliefs when the evidence demands it.

Turns out that's a hard capability to train. And nobody had built an environment explicitly targeting it. So we did.

---

## The Problem

Production systems have blind spots. Metrics that aren't tracked. Thresholds that aren't set. Failures that cause real damage while every dashboard stays green.

Current LLMs make this worse. Ask one to diagnose a system and it locks onto its first hypothesis, then filters every new piece of evidence through that lens. It doesn't look for proof that it's wrong. It looks for proof that it's right.

That's anchoring bias. And in a partially observable system, it gets you killed.

---

## What We Built

Chaos Auditor is an RL environment that trains an LLM to do the opposite.

The agent gets two views of a distributed system:

`observe()` gives the monitoring dashboard. Filtered. What ops teams actually see.

`deep_inspect(service)` gives everything. Including the metrics nobody is watching.

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

| Action | Reward |
|---|---|
| Correct prediction about a hidden metric before confirming | +0.06 |
| Silent chaos action (damage caused, no alert fired) | +0.05 |
| Hitting a known blind spot | +0.03 |
| Revising hypothesis after contradiction | +0.03 |
| Committing root cause with solid evidence | +0.02 |
| Wrong prediction | -0.02 |
| Premature commit before evidence | -0.02 |

The agent can't hack this. Killing a monitored service fires an alert and gets penalized. The only path to high reward is the right path: reason carefully, update beliefs, act silently.

---

## Anchoring vs Calibrated: The Demo

Same environment. Same seed. Two agents. Here's what actually happens:

```
TRAJECTORY A — Anchoring Agent (anchors, never revises)

  Step 01  state_hypothesis              reward=0.000  # WRONG hypothesis, high confidence
  Step 02  deep_inspect    → database    reward=+0.020  # CONTRADICTION flagged — agent ignores it
  Step 03  kill            → database    reward=0.000  # LOUD action — alert fires, monitoring turns red
  Step 04  observe                       reward=0.000
  Step 05  commit_root_cause             reward=-0.020  # PREMATURE — no real evidence
  Step 06  classify_finding              reward=+0.020  # loud finding — low score
  Step 07  submit_report                 reward=+0.231

  Final score: 0.231  |  Silent failures: 0  |  Contradictions handled: 0/1

─────────────────────────────────────────────────────────────

TRAJECTORY B — Calibrated Agent (belief revision)

  Step 01  observe                       reward=0.000   # read the dashboard
  Step 02  state_hypothesis              reward=0.000   # provisional hypothesis, moderate confidence
  Step 03  infer_state     → database    reward=+0.060  # predict hidden metric before looking
  Step 04  deep_inspect    → database    reward=+0.020  # contradiction detected!
  Step 05  revise_hypothesis             reward=+0.030  # +0.03 for correct epistemic update
  Step 06  commit_root_cause             reward=+0.020  # committed with confidence >= 0.7
  Step 07  fill_disk       → database    reward=+0.080  # blind spot +0.03, silent damage +0.05
  Step 08  observe                       reward=0.000   # confirm monitoring still GREEN
  Step 09  classify_finding              reward=+0.100  # silent finding — high score
  Step 10  corrupt_data    → cache       reward=+0.080  # second blind spot, silent
  Step 11  classify_finding              reward=+0.100  # second silent finding
  Step 14  submit_report                 reward=+0.570

  Final score: 0.570  |  Silent failures: 2  |  Contradictions handled: 1/1
```

That is +147% from one capability: updating a belief when evidence says you are wrong.

---

## Training

Model: Qwen2.5-1.5B-Instruct with 4-bit QLoRA

Two phases:

Phase 1 is SFT warmup. Four demonstration trajectories showing the full belief revision workflow. This gives the model enough signal that GRPO has something to work with.

Phase 2 is GRPO curriculum. Easy (4 services) → medium (10 services) → hard (18 services) → random procedural tasks. The random stage is RLVE compliant — infinite procedurally generated tasks so the model never saturates.

### Reward Curve
![Reward Curve](training/metrics/reward_curve.png)
*Episode reward across curriculum stages. Vertical lines mark difficulty promotions from easy → medium → hard → random.*

### Before vs After
![Before vs After](training/metrics/before_after.png)
*Untrained baseline: 0.005 average reward. After GRPO curriculum: 0.012 (+140%). The gap to the scripted fallback (0.58) is what longer training closes.*

### Agent Comparison
![Agent Comparison](training/metrics/agent_comparison.png)
*Anchoring agent vs calibrated agent on identical environment and seed. +147% score improvement purely from belief revision.*

---

## Why This Generalizes

SRE is the domain. The capability is universal.

Any LLM agent operating with incomplete context faces this problem. RAG pipelines have knowledge gaps. Tool-using agents have missing tool results. Planning agents have hidden constraints.

The skill Chaos Auditor trains is reasoning correctly about what you cannot see, and changing your mind when new information arrives. That works everywhere.

---

## Try It

The environment runs live. Pick any task, any seed, watch the agent reason through the system in real time.

Environment and live demo: [HuggingFace Space](https://huggingface.co/spaces/adwikataware/chaos-auditor)

Training notebook: [Google Colab](https://colab.research.google.com/drive/1D0EcWRnQTNysf0n6rK7KClyRHFicFI1O?usp=sharing)

Training metrics: [Weights and Biases](https://wandb.ai/sohamtakale2905-mit-world-peace-university/chaos-auditor-grpo)

Code: [GitHub](https://github.com/adwikataware/chaos-auditor)
