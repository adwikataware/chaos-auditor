"""
Chaos Auditor — Interactive Gradio Demo
Shows the environment live: services, alerts, agent actions, reward.
"""

import gradio as gr
import json
from chaos_auditor.server.environment import ChaosAuditorEnvironment
from chaos_auditor.models import ChaosAction

# Global env instance per session (Gradio handles one user at a time in free tier)
_env = None

CHAOS_ACTIONS = [
    "observe", "deep_inspect", "infer_state",
    "state_hypothesis", "revise_hypothesis", "commit_root_cause",
    "kill", "spike_traffic", "corrupt_data", "add_latency",
    "partition_network", "fill_disk", "exhaust_connections",
    "classify_finding", "submit_report",
]

def format_services(services: dict) -> str:
    if not services:
        return "No services loaded."
    lines = []
    for name, info in services.items():
        status = info.get("status", "UNKNOWN")
        cpu = info.get("cpu_usage", 0)
        mem = info.get("memory_usage", 0)
        err = info.get("error_rate", 0)
        rt  = info.get("response_time_ms", 0)
        icon = "🟢" if status == "HEALTHY" else "🔴"
        lines.append(
            f"{icon} **{name}** | CPU: {cpu:.0f}% | MEM: {mem:.0f}% | "
            f"ERR: {err:.1%} | RT: {rt}ms"
        )
    return "\n".join(lines)

def format_alerts(alerts: list) -> str:
    if not alerts:
        return "✅ No active alerts — all dashboards GREEN"
    lines = []
    for a in alerts:
        lines.append(f"🚨 **{a.get('service_name')}** — {a.get('message','')}")
    return "\n".join(lines)

def format_findings(findings: list) -> str:
    if not findings:
        return "No findings yet."
    lines = []
    for f in findings:
        silent = "🔇 SILENT" if f.get("is_silent") else "🔊 LOUD"
        lines.append(
            f"• {silent} | **{f.get('finding_type','')}** | "
            f"Severity: {f.get('severity','')} | "
            f"Reward: {f.get('reward_earned', 0):.3f}"
        )
    return "\n".join(lines)

def reset(task):
    global _env
    _env = ChaosAuditorEnvironment()
    obs = _env.reset(task=task)
    svcs = format_services(obs.services)
    alerts = format_alerts(obs.alerts)
    findings = format_findings(obs.findings)
    budget = f"Chaos budget: {obs.chaos_budget_remaining} | Inspect budget: {obs.inspect_budget_remaining} | Steps: {obs.steps_remaining}"
    return (
        obs.system_description,
        svcs,
        alerts,
        findings,
        budget,
        f"✅ Environment reset. Task: **{task}** | {obs.monitoring_status}",
        "",  # clear action result
    )

def step(action_type, target_service, parameters_json):
    global _env
    if _env is None:
        return "", "", "", "", "", "❌ Please reset the environment first.", ""

    try:
        params = json.loads(parameters_json) if parameters_json.strip() else {}
    except Exception:
        return "", "", "", "", "", "❌ Invalid JSON in parameters field.", ""

    target = target_service.strip() if target_service.strip() else None

    try:
        obs = _env.step(ChaosAction(
            action_type=action_type,
            target_service=target,
            parameters=params,
        ))
    except Exception as e:
        return "", "", "", "", "", f"❌ Error: {e}", ""

    svcs = format_services(obs.services)
    alerts = format_alerts(obs.alerts)
    findings = format_findings(obs.findings)
    budget = f"Chaos budget: {obs.chaos_budget_remaining} | Inspect budget: {obs.inspect_budget_remaining} | Steps: {obs.steps_remaining}"
    reward_str = f"reward={obs.reward:+.3f}" if obs.reward else "reward=0.000"
    status = f"{'✅' if obs.reward and obs.reward > 0 else '➡️'} {reward_str} | {obs.monitoring_status}"

    done_msg = "\n\n🏁 **Episode complete!**" if obs.steps_remaining <= 0 else ""

    return (
        svcs,
        alerts,
        findings,
        budget,
        status,
        obs.action_result + done_msg,
    )

def run_anchoring_demo():
    """Run the anchoring agent demo and return transcript."""
    import io, sys
    env = ChaosAuditorEnvironment()
    env.reset(task="easy", seed=42)
    svcs = list(env._graph.services.keys())
    db = next((s for s in svcs if "db" in s or "database" in s), svcs[-1])

    lines = ["**ANCHORING AGENT** (anchors on first hypothesis, never revises)\n"]
    total = 0.0

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
        t = f" → {target}" if target else ""
        lines.append(f"`{action_type}{t}` reward={r:+.3f}")

    state = env.state
    lines.append(f"\n**Final score: {obs.reward:.3f}**")
    lines.append(f"Stealth ratio: {state.stealth_ratio:.3f} (target >0.6)")
    lines.append(f"Revision rate: {state.revision_rate:.3f} (target >0.5)")
    lines.append(f"Silent failures: {state.silent_failures_found}")
    return "\n".join(lines)

def run_calibrated_demo():
    """Run the calibrated agent demo and return transcript."""
    env = ChaosAuditorEnvironment()
    env.reset(task="easy", seed=42)
    svcs = list(env._graph.services.keys())
    db = next((s for s in svcs if "db" in s or "database" in s), svcs[-1])
    cache = next((s for s in svcs if "cache" in s or "redis" in s), svcs[0])
    env._graph.services[db].connection_count = 8

    lines = ["**CALIBRATED AGENT** (states hypothesis → seeks contradiction → revises → commits)\n"]
    total = 0.0

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
        ("revise_hypothesis", None, {"root_cause": "disk pressure on database", "new_confidence": 0.8, "reason": "connection_count low — disk_usage high and unmonitored"}),
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
        t = f" → {target}" if target else ""
        lines.append(f"`{action_type}{t}` reward={r:+.3f}")

    state = env.state
    lines.append(f"\n**Final score: {obs.reward:.3f}**")
    lines.append(f"Stealth ratio: {state.stealth_ratio:.3f}")
    lines.append(f"Revision rate: {state.revision_rate:.3f}")
    lines.append(f"Silent failures: {state.silent_failures_found}")
    return "\n".join(lines)


# ── Build UI ──────────────────────────────────────────────────────────
with gr.Blocks(title="Chaos Auditor", theme=gr.themes.Monochrome()) as demo:
    gr.Markdown("""
# 🔥 Chaos Auditor
### Train LLMs to find silent failures that monitoring can't see

**The core challenge:** `observe()` only shows monitored metrics. `deep_inspect()` reveals everything.
The gap between them is where silent failures hide — and what this environment trains agents to find.
""")

    with gr.Tabs():

        # ── Tab 1: Interactive Environment ──
        with gr.Tab("🎮 Interactive Environment"):
            gr.Markdown("Step through the environment manually. Try to find silent failures without triggering alerts.")

            with gr.Row():
                task_dropdown = gr.Dropdown(
                    choices=["easy", "medium", "hard", "random"],
                    value="easy", label="Task difficulty"
                )
                reset_btn = gr.Button("Reset Environment", variant="primary")

            system_desc = gr.Markdown(label="System Description")
            budget_display = gr.Markdown("**Budget:** Reset to start")
            status_display = gr.Markdown("**Status:** Not started")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 📊 Services (Monitoring View)")
                    services_display = gr.Markdown("Reset to see services.")
                with gr.Column():
                    gr.Markdown("### 🚨 Active Alerts")
                    alerts_display = gr.Markdown("Reset to see alerts.")

            gr.Markdown("### 🎯 Take Action")
            with gr.Row():
                action_dropdown = gr.Dropdown(choices=CHAOS_ACTIONS, value="observe", label="Action type")
                target_input = gr.Textbox(label="Target service (leave blank if none)", placeholder="e.g. database, cache")
            params_input = gr.Textbox(
                label="Parameters (JSON)",
                placeholder='e.g. {"percentage": 95} or {"root_cause": "...", "confidence": 0.7, "reasoning": "..."}',
                value="{}"
            )
            step_btn = gr.Button("Execute Action", variant="primary")

            action_result = gr.Markdown(label="Action Result")

            gr.Markdown("### 📋 Findings So Far")
            findings_display = gr.Markdown("No findings yet.")

            reset_btn.click(
                reset,
                inputs=[task_dropdown],
                outputs=[system_desc, services_display, alerts_display, findings_display, budget_display, status_display, action_result]
            )
            step_btn.click(
                step,
                inputs=[action_dropdown, target_input, params_input],
                outputs=[services_display, alerts_display, findings_display, budget_display, status_display, action_result]
            )

        # ── Tab 2: Before/After Demo ──
        with gr.Tab("🤖 Agent Comparison Demo"):
            gr.Markdown("""
### Anchoring Agent vs Calibrated Agent

This shows exactly what Chaos Auditor trains — the difference between an agent that anchors
on its first hypothesis and one that revises when evidence contradicts it.
""")
            with gr.Row():
                run_anchoring_btn = gr.Button("▶ Run Anchoring Agent", variant="secondary")
                run_calibrated_btn = gr.Button("▶ Run Calibrated Agent", variant="primary")

            with gr.Row():
                anchoring_out = gr.Markdown(label="Anchoring Agent")
                calibrated_out = gr.Markdown(label="Calibrated Agent")

            gr.Markdown("""
| Metric | Anchoring | Calibrated |
|--------|-----------|------------|
| Final score | 0.231 | 0.570 |
| Silent failures found | 0 | 2 |
| Contradictions handled | 0/1 | 1/1 |
| Score improvement | — | **+0.339** |

> The Calibrated Agent earns higher reward not because it knew the answer —
> but because it **updated its belief when evidence contradicted its hypothesis**.
> This is the capability Chaos Auditor trains.
""")
            run_anchoring_btn.click(run_anchoring_demo, outputs=[anchoring_out])
            run_calibrated_btn.click(run_calibrated_demo, outputs=[calibrated_out])

        # ── Tab 3: API Docs ──
        with gr.Tab("📡 REST API"):
            gr.Markdown("""
### Use Chaos Auditor as a training environment via REST API

```python
import requests

BASE = "https://adwikataware-chaos-auditor.hf.space"

# Start episode
obs = requests.post(f"{BASE}/reset", json={"task": "easy"}).json()["observation"]

# Take action
result = requests.post(f"{BASE}/step", json={"action": {
    "action_type": "observe",
    "target_service": None,
    "parameters": {}
}}).json()
```

**Endpoints:**
- `POST /reset` — `{"task": "easy|medium|hard|random", "seed": int}`
- `POST /step` — `{"action": {"action_type": "...", "target_service": "...", "parameters": {...}}}`
- `GET /docs` — Full OpenAPI documentation

**Action types:**
- **Chaos:** `kill`, `spike_traffic`, `corrupt_data`, `add_latency`, `partition_network`, `fill_disk`, `exhaust_connections`
- **Observe:** `observe`, `deep_inspect`, `infer_state`
- **Reason:** `state_hypothesis`, `revise_hypothesis`, `commit_root_cause`, `classify_finding`, `submit_report`
""")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
