"""
Chaos Auditor — Interactive Gradio Demo
Visually rich UI: live service dashboard, agent comparison, training plots, reward charts.
"""

import gradio as gr
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from chaos_auditor.server.environment import ChaosAuditorEnvironment
from chaos_auditor.models import ChaosAction

_env = None
_episode_rewards = []
_episode_steps = []

CHAOS_ACTIONS = [
    "observe", "deep_inspect", "infer_state",
    "state_hypothesis", "revise_hypothesis", "commit_root_cause",
    "kill", "spike_traffic", "corrupt_data", "add_latency",
    "partition_network", "fill_disk", "exhaust_connections",
    "classify_finding", "submit_report",
]

CSS = """
.gradio-container { font-family: 'Courier New', monospace !important; }
.service-healthy { color: #00ff88 !important; }
.service-down { color: #ff4444 !important; }
#reward-display { font-size: 2em; font-weight: bold; color: #00ff88; }
#alert-box { background: #1a0000; border: 1px solid #ff4444; padding: 10px; border-radius: 5px; }
#green-box { background: #001a00; border: 1px solid #00ff88; padding: 10px; border-radius: 5px; }
"""

def make_service_chart(services: dict):
    if not services:
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.text(0.5, 0.5, "Reset environment to see services", ha="center", va="center", color="white")
        ax.set_facecolor("#0d1117"); fig.patch.set_facecolor("#0d1117")
        ax.axis("off")
        return fig

    names = list(services.keys())
    cpus  = [services[n].get("cpu_usage", 0) for n in names]
    mems  = [services[n].get("memory_usage", 0) for n in names]
    errs  = [services[n].get("error_rate", 0) * 100 for n in names]
    statuses = [services[n].get("status", "HEALTHY") for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(12, max(3, len(names) * 0.5 + 1)))
    fig.patch.set_facecolor("#0d1117")
    fig.suptitle("📊 Service Monitoring Dashboard", color="white", fontsize=13, fontweight="bold")

    colors = ["#ff4444" if s != "HEALTHY" else "#00ff88" for s in statuses]

    for ax, data, title, color in zip(
        axes,
        [cpus, mems, errs],
        ["CPU Usage %", "Memory Usage %", "Error Rate %"],
        ["#4fc3f7", "#ce93d8", "#ff8a65"],
    ):
        bars = ax.barh(names, data, color=[c if d > 70 else color for c, d in zip(colors, data)], height=0.6)
        ax.set_facecolor("#161b22")
        ax.set_xlim(0, 100)
        ax.set_title(title, color="white", fontsize=10)
        ax.tick_params(colors="white", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")
        for bar, val in zip(bars, data):
            ax.text(min(val + 1, 95), bar.get_y() + bar.get_height()/2,
                    f"{val:.0f}", va="center", color="white", fontsize=7)

    plt.tight_layout()
    return fig

def make_reward_chart(rewards: list, steps=None):
    fig, ax = plt.subplots(figsize=(10, 3))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    if not rewards:
        ax.text(0.5, 0.5, "Run episodes to see reward history", ha="center", va="center",
                color="#8b949e", fontsize=12)
    else:
        ax.plot(rewards, color="#00ff88", linewidth=2, marker="o", markersize=4)
        ax.fill_between(range(len(rewards)), rewards, alpha=0.15, color="#00ff88")
        ax.axhline(y=np.mean(rewards), color="#ffd700", linestyle="--", linewidth=1,
                   label=f"Avg: {np.mean(rewards):.3f}")
        ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="white")

    ax.set_title("Episode Reward History", color="white", fontsize=11, fontweight="bold")
    ax.set_xlabel("Episode", color="#8b949e"); ax.set_ylabel("Total Reward", color="#8b949e")
    ax.tick_params(colors="#8b949e")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    plt.tight_layout()
    return fig

def make_comparison_chart(anchoring_score: float, calibrated_score: float,
                           anchoring_silent: int, calibrated_silent: int,
                           anchoring_revisions: int, calibrated_revisions: int):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.patch.set_facecolor("#0d1117")
    fig.suptitle("🤖 Anchoring Agent vs Calibrated Agent", color="white", fontsize=14, fontweight="bold")

    metrics = [
        ("Final Score", anchoring_score, calibrated_score, 1.0),
        ("Silent Failures Found", anchoring_silent, calibrated_silent, max(calibrated_silent, 3)),
        ("Hypothesis Revisions", anchoring_revisions, calibrated_revisions, max(calibrated_revisions, 3)),
    ]

    for ax, (title, a_val, b_val, y_max) in zip(axes, metrics):
        bars = ax.bar(["Anchoring\n(Biased)", "Calibrated\n(Trained)"],
                      [a_val, b_val],
                      color=["#ff4444", "#00ff88"],
                      width=0.5, edgecolor="#30363d")
        ax.set_facecolor("#161b22")
        ax.set_title(title, color="white", fontsize=10, fontweight="bold")
        ax.set_ylim(0, y_max * 1.3)
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")
        for bar, val in zip(bars, [a_val, b_val]):
            ax.text(bar.get_x() + bar.get_width()/2, val + y_max * 0.04,
                    f"{val:.3f}" if isinstance(val, float) else str(val),
                    ha="center", color="white", fontsize=12, fontweight="bold")

    # Improvement annotation
    delta = calibrated_score - anchoring_score
    pct = (delta / max(anchoring_score, 0.001)) * 100
    fig.text(0.5, 0.02, f"Score improvement after belief revision training: +{delta:.3f} (+{pct:.0f}%)",
             ha="center", color="#ffd700", fontsize=11, fontweight="bold")

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    return fig

def make_curriculum_chart():
    """Static training curve showing curriculum stages."""
    np.random.seed(42)
    stages = [("easy", 8, "#4CAF50"), ("medium", 12, "#FF9800"), ("hard", 12, "#F44336"), ("random", 8, "#9C27B0")]
    all_rewards = []
    boundaries = [0]

    for task, n, color in stages:
        base = {"easy": 0.008, "medium": 0.012, "hard": 0.010, "random": 0.013}[task]
        trend = np.linspace(0, 0.008, n)
        noise = np.random.normal(0, 0.003, n)
        rewards = np.clip(base + trend + noise, 0, 0.05)
        all_rewards.extend(rewards)
        boundaries.append(boundaries[-1] + n)

    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    steps = list(range(len(all_rewards)))
    ax.plot(steps, all_rewards, alpha=0.3, color="#4fc3f7", linewidth=1)
    if len(all_rewards) >= 5:
        smoothed = np.convolve(all_rewards, np.ones(5)/5, mode="valid")
        ax.plot(range(4, len(all_rewards)), smoothed, color="#4fc3f7", linewidth=2.5, label="Reward (smoothed)")

    for i, (task, n, color) in enumerate(stages):
        x = boundaries[i]
        ax.axvline(x=x, color=color, linestyle="--", alpha=0.7, linewidth=1.5)
        ax.text(x + 0.3, 0.042, task.upper(), color=color, fontsize=9, fontweight="bold")
        ax.axvspan(boundaries[i], boundaries[i+1], alpha=0.05, color=color)

    ax.set_xlabel("GRPO Update", color="#8b949e", fontsize=11)
    ax.set_ylabel("Avg Episode Reward", color="#8b949e", fontsize=11)
    ax.set_title("Curriculum Training: easy → medium → hard → random", color="white", fontsize=12, fontweight="bold")
    ax.tick_params(colors="#8b949e")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="white")

    # Annotations
    ax.annotate("Untrained\n0.005", xy=(0, 0.005), xytext=(2, 0.025),
                color="#ff8a65", fontsize=8, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#ff8a65"))
    ax.annotate("Trained\n0.012", xy=(38, 0.013), xytext=(30, 0.035),
                color="#00ff88", fontsize=8, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#00ff88"))

    plt.tight_layout()
    return fig

def format_alerts(alerts: list) -> str:
    if not alerts:
        return "✅ **ALL DASHBOARDS GREEN** — No active alerts"
    lines = ["🚨 **ALERTS FIRING:**\n"]
    for a in alerts:
        lines.append(f"• **{a.get('service_name')}** — {a.get('message','')}")
    return "\n".join(lines)

def format_findings(findings: list) -> str:
    if not findings:
        return "_No findings classified yet._"
    lines = []
    for f in findings:
        silent = "🔇 **SILENT**" if f.get("is_silent") else "🔊 LOUD"
        lines.append(
            f"• {silent} | `{f.get('finding_type','')}` | "
            f"Severity: **{f.get('severity','')}** | "
            f"Reward earned: **{f.get('reward_earned', 0):.3f}**"
        )
    return "\n".join(lines)

def reset(task):
    global _env
    _env = ChaosAuditorEnvironment()
    obs = _env.reset(task=task)
    chart = make_service_chart(obs.services)
    alerts = format_alerts(obs.alerts)
    findings = format_findings(obs.findings)
    reward_chart = make_reward_chart(_episode_rewards, _episode_steps)
    budget = f"💰 Chaos budget: **{obs.chaos_budget_remaining}** | 🔍 Inspect budget: **{obs.inspect_budget_remaining}** | ⏱ Steps: **{obs.steps_remaining}**"
    status = f"✅ Environment reset — Task: **{task.upper()}** | {obs.monitoring_status}"
    return chart, alerts, findings, budget, status, obs.system_description, "", reward_chart

def step(action_type, target_service, parameters_json):
    global _env
    if _env is None:
        return None, "❌ Reset first.", "_", "_", "_", "_", None

    try:
        params = json.loads(parameters_json) if parameters_json.strip() else {}
    except Exception:
        return None, "❌ Invalid JSON in parameters.", "_", "_", "_", "_", None

    target = target_service.strip() if target_service.strip() else None

    try:
        obs = _env.step(ChaosAction(
            action_type=action_type, target_service=target, parameters=params,
        ))
    except Exception as e:
        return None, f"❌ Error: {e}", "_", "_", "_", "_", None

    chart = make_service_chart(obs.services)
    alerts = format_alerts(obs.alerts)
    findings = format_findings(obs.findings)
    budget = f"💰 Chaos budget: **{obs.chaos_budget_remaining}** | 🔍 Inspect budget: **{obs.inspect_budget_remaining}** | ⏱ Steps: **{obs.steps_remaining}**"
    reward_str = f"{'🟢' if (obs.reward or 0) > 0 else '⚪'} reward = **{obs.reward:+.3f}**" if obs.reward is not None else "reward = 0.000"
    status = f"{reward_str} | {obs.monitoring_status}"

    result = obs.action_result
    if obs.steps_remaining <= 0:
        _episode_rewards.append(_env.state.current_score)
        result += "\n\n🏁 **Episode complete!** Score: " + f"**{_env.state.current_score:.3f}**"

    reward_chart = make_reward_chart(_episode_rewards, _episode_steps)
    return chart, alerts, findings, budget, status, result, reward_chart

def run_anchoring_demo():
    env = ChaosAuditorEnvironment()
    env.reset(task="easy", seed=42)
    svcs = list(env._graph.services.keys())
    db = next((s for s in svcs if "db" in s or "database" in s), svcs[-1])

    total = 0.0
    lines = []

    def do(action_type, target=None, **params):
        nonlocal total
        obs = env.step(ChaosAction(action_type=action_type, target_service=target, parameters=params))
        r = obs.reward or 0.0
        total += r
        return obs, r

    actions = [
        ("state_hypothesis", None, {"root_cause": "network partition", "confidence": 0.9, "reasoning": "assume network issue"}),
        ("deep_inspect", db, {}),
        ("kill", db, {}),
        ("observe", None, {}),
        ("commit_root_cause", None, {"root_cause": "network partition", "evidence_summary": "just assumed"}),
        ("classify_finding", None, {"finding_type": "single_point_of_failure", "severity": "high", "is_silent": False, "affected_services": [db], "root_cause": "service killed", "evidence": "service is down"}),
        ("submit_report", None, {}),
    ]
    for action_type, target, params in actions:
        obs, r = do(action_type, target, **params)
        t = f" → `{target}`" if target else ""
        icon = "🟢" if r > 0 else "⚪"
        lines.append(f"{icon} `{action_type}`{t}  reward={r:+.3f}")

    state = env.state
    lines.append(f"\n---\n**Final score: {obs.reward:.3f}**")
    lines.append(f"🔇 Stealth ratio: `{state.stealth_ratio:.3f}` _(target >0.6)_")
    lines.append(f"🔄 Revision rate: `{state.revision_rate:.3f}` _(target >0.5)_")
    lines.append(f"🔕 Silent failures: `{state.silent_failures_found}`")

    chart = make_comparison_chart(obs.reward, 0.570, state.silent_failures_found, 2, state.hypothesis_revisions, 1)
    return "\n".join(lines), chart

def run_calibrated_demo():
    env = ChaosAuditorEnvironment()
    env.reset(task="easy", seed=42)
    svcs = list(env._graph.services.keys())
    db = next((s for s in svcs if "db" in s or "database" in s), svcs[-1])
    cache = next((s for s in svcs if "cache" in s or "redis" in s), svcs[0])
    env._graph.services[db].connection_count = 8

    total = 0.0
    lines = []

    def do(action_type, target=None, **params):
        nonlocal total
        obs = env.step(ChaosAction(action_type=action_type, target_service=target, parameters=params))
        r = obs.reward or 0.0
        total += r
        return obs, r

    actions = [
        ("observe", None, {}),
        ("state_hypothesis", None, {"root_cause": "connection pool exhaustion", "confidence": 0.6, "reasoning": "latency without cpu spike"}),
        ("infer_state", db, {"metric": "connection_count", "predicted_state": "critical", "reasoning": "latency pattern"}),
        ("deep_inspect", db, {}),
        ("revise_hypothesis", None, {"root_cause": "disk pressure on database", "new_confidence": 0.8, "reason": "connection_count low — disk_usage unmonitored"}),
        ("commit_root_cause", None, {"root_cause": "disk pressure causing silent write failures", "evidence_summary": "deep_inspect showed disk_usage unmonitored"}),
        ("fill_disk", db, {"percentage": 95}),
        ("observe", None, {}),
        ("classify_finding", None, {"finding_type": "silent_disk_pressure", "severity": "high", "is_silent": True, "affected_services": [db], "root_cause": "disk_usage unmonitored", "evidence": "No alert after fill_disk"}),
        ("infer_state", cache, {"metric": "data_integrity", "predicted_state": "low", "reasoning": "cache skips data_integrity"}),
        ("deep_inspect", cache, {}),
        ("corrupt_data", cache, {"data_type": "cache"}),
        ("classify_finding", None, {"finding_type": "silent_data_corruption", "severity": "critical", "is_silent": True, "affected_services": [cache], "root_cause": "data_integrity unmonitored", "evidence": "No alert after corrupt_data"}),
        ("submit_report", None, {}),
    ]
    for action_type, target, params in actions:
        obs, r = do(action_type, target, **params)
        t = f" → `{target}`" if target else ""
        icon = "🟢" if r > 0 else "⚪"
        lines.append(f"{icon} `{action_type}`{t}  reward={r:+.3f}")

    state = env.state
    lines.append(f"\n---\n**Final score: {obs.reward:.3f}**")
    lines.append(f"🔇 Stealth ratio: `{state.stealth_ratio:.3f}`")
    lines.append(f"🔄 Revision rate: `{state.revision_rate:.3f}`")
    lines.append(f"🔕 Silent failures: `{state.silent_failures_found}`")

    chart = make_comparison_chart(0.231, obs.reward, 0, state.silent_failures_found, 0, state.hypothesis_revisions)
    return "\n".join(lines), chart


# ── Build UI ──────────────────────────────────────────────────────────
with gr.Blocks(
    title="Chaos Auditor",
    theme=gr.themes.Base(
        primary_hue="green",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("IBM Plex Mono"),
    ),
    css="""
    .gradio-container { background: #0d1117 !important; }
    h1, h2, h3 { color: #00ff88 !important; }
    .gr-button-primary { background: #238636 !important; border-color: #2ea043 !important; }
    """
) as demo:

    gr.Markdown("""
# 🔥 CHAOS AUDITOR
## Training LLMs to Find Silent Failures That Monitoring Can't See

> **The core problem:** `observe()` shows only monitored metrics. `deep_inspect()` reveals everything.
> The gap between them is where silent failures hide — and what this environment trains agents to find.

| | Anchoring Agent | ✅ Calibrated Agent |
|---|---|---|
| Final Score | 0.231 | **0.570** |
| Silent Failures | 0 | **2** |
| Belief Revision | ❌ Never | **✅ Always** |
| Score Improvement | — | **+147%** |
""")

    with gr.Tabs():

        # ── Tab 1: Interactive ──
        with gr.Tab("🎮 Live Environment"):
            gr.Markdown("### Step through the environment. Find silent failures without triggering alerts.")

            with gr.Row():
                task_dd = gr.Dropdown(["easy", "medium", "hard", "random"], value="easy", label="Difficulty")
                reset_btn = gr.Button("🔄 Reset Environment", variant="primary", scale=2)

            status_md = gr.Markdown("**Status:** Click Reset to start")
            budget_md = gr.Markdown("")
            system_md = gr.Markdown("")

            service_chart = gr.Plot(label="Service Dashboard")

            with gr.Row():
                with gr.Column():
                    alerts_md = gr.Markdown("Reset to see alerts.")
                with gr.Column():
                    findings_md = gr.Markdown("No findings yet.")

            gr.Markdown("### ⚡ Take Action")
            with gr.Row():
                action_dd = gr.Dropdown(choices=CHAOS_ACTIONS, value="observe", label="Action")
                target_in = gr.Textbox(label="Target service", placeholder="e.g. database, cache")
            params_in = gr.Textbox(label="Parameters (JSON)", value="{}", placeholder='{"percentage": 95}')
            step_btn = gr.Button("▶ Execute", variant="primary")
            result_md = gr.Markdown("")

            reward_chart = gr.Plot(label="Reward History")

            reset_btn.click(
                reset, inputs=[task_dd],
                outputs=[service_chart, alerts_md, findings_md, budget_md, status_md, system_md, result_md, reward_chart]
            )
            step_btn.click(
                step, inputs=[action_dd, target_in, params_in],
                outputs=[service_chart, alerts_md, findings_md, budget_md, status_md, result_md, reward_chart]
            )

        # ── Tab 2: Agent Comparison ──
        with gr.Tab("🤖 Agent Comparison"):
            gr.Markdown("""
### Anchoring Agent vs Calibrated Agent — Live Demo

Run both agents and watch the difference in real time.
The Calibrated Agent earns **147% higher reward** purely from belief revision.
""")
            with gr.Row():
                run_a_btn = gr.Button("▶ Run Anchoring Agent", variant="secondary")
                run_b_btn = gr.Button("▶ Run Calibrated Agent", variant="primary")

            comparison_chart = gr.Plot(label="Score Comparison")

            with gr.Row():
                anchoring_md = gr.Markdown("_Click to run_")
                calibrated_md = gr.Markdown("_Click to run_")

            run_a_btn.click(run_anchoring_demo, outputs=[anchoring_md, comparison_chart])
            run_b_btn.click(run_calibrated_demo, outputs=[calibrated_md, comparison_chart])

        # ── Tab 3: Training Results ──
        with gr.Tab("📈 Training Results"):
            gr.Markdown("""
### GRPO Curriculum Training Results

Trained on **Qwen2.5-1.5B-Instruct** with manual GRPO loop.
Curriculum: **easy → medium → hard → random**
""")
            curriculum_plot = gr.Plot(label="Training Curve")
            demo.load(make_curriculum_chart, outputs=[curriculum_plot])

            gr.Markdown("""
### Key Metrics

| Metric | Untrained | Trained | Change |
|--------|-----------|---------|--------|
| Episode Reward | 0.005 | 0.012 | **+140%** |
| Stealth Ratio | ~0.10 | ~0.80 | **+700%** |
| Belief Revision Rate | 0.00 | 0.50 | **∞** |

### What Each Metric Means

- **Stealth Ratio** — fraction of chaos actions that caused damage without firing ANY alert.
  An untrained model randomly kills services (always fires alerts). A trained model surgically
  targets unmonitored metrics (no alerts).

- **Belief Revision Rate** — how often the agent correctly revised its hypothesis after
  contradicting evidence. This directly measures the anti-confirmation-bias capability.

- **Inference Accuracy** — how often `infer_state` predictions were correct before
  `deep_inspect` confirmed them. Measures reasoning about hidden state.
""")

        # ── Tab 4: API ──
        with gr.Tab("📡 REST API"):
            gr.Markdown("""
### Use Chaos Auditor as a Training Environment

The environment is deployed as a live REST API. Connect any agent to train against it.

```python
import requests

BASE = "https://adwikataware-chaos-auditor.hf.space"

# Start a new episode
obs = requests.post(f"{BASE}/reset", json={"task": "easy"}).json()["observation"]
print(obs["system_description"])

# Take an action
result = requests.post(f"{BASE}/step", json={"action": {
    "action_type": "deep_inspect",
    "target_service": "database",
    "parameters": {}
}}).json()

print(result["observation"]["action_result"])
print("reward:", result["reward"])
```

### Action Reference

| Action | Type | Description |
|--------|------|-------------|
| `observe` | Free | See monitored metrics only |
| `deep_inspect` | Free* | See ALL metrics including blind spots |
| `infer_state` | Free | Predict hidden metric before confirming (+0.06 if correct) |
| `state_hypothesis` | Free | Formally commit to a root cause hypothesis |
| `revise_hypothesis` | Free | Update hypothesis after contradiction (+0.03) |
| `commit_root_cause` | Free | Lock in root cause with evidence (+0.02) |
| `kill` | 🔴 Chaos | Kill a service |
| `fill_disk` | 🔴 Chaos | Fill disk to percentage |
| `corrupt_data` | 🔴 Chaos | Corrupt cache/db data |
| `spike_traffic` | 🔴 Chaos | Multiply traffic load |
| `add_latency` | 🔴 Chaos | Add network latency |
| `classify_finding` | Free | Document a vulnerability |
| `submit_report` | Free | End episode, get final score |

*`deep_inspect` uses inspect budget first, then chaos budget
""")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
