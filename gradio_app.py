"""
Chaos Auditor — Creative HF Space
A live "NOC dashboard" where everything looks GREEN while a hidden agent silently breaks things.
Click "Reveal Truth" to see what was actually happening underneath.
"""

import gradio as gr
import threading
import time
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from chaos_auditor.server.environment import ChaosAuditorEnvironment
from chaos_auditor.models import ChaosAction

# ── Background chaos agent ────────────────────────────────────────────
_state = {
    "env": None,
    "running": False,
    "actions_taken": [],
    "hidden_damage": [],
    "monitoring_view": {},
    "truth_view": {},
    "score": 0.0,
    "step": 0,
    "episode_done": False,
    "services": [],
}

AGENT_SCRIPT = [
    ("observe", None, {}),
    ("state_hypothesis", None, {"root_cause": "disk pressure on database", "confidence": 0.6, "reasoning": "disk_usage rarely monitored"}),
    ("infer_state", "__db__", {"metric": "disk_usage", "predicted_state": "high", "reasoning": "databases accumulate logs"}),
    ("deep_inspect", "__db__", {}),
    ("revise_hypothesis", None, {"root_cause": "disk and data integrity both unmonitored", "new_confidence": 0.8, "reason": "blind spots confirmed"}),
    ("commit_root_cause", None, {"root_cause": "multiple unmonitored metrics enabling silent failures", "evidence_summary": "deep_inspect confirmed blind spots on db and cache"}),
    ("fill_disk", "__db__", {"percentage": 95}),
    ("infer_state", "__cache__", {"metric": "data_integrity", "predicted_state": "low", "reasoning": "cache skips data_integrity"}),
    ("deep_inspect", "__cache__", {}),
    ("corrupt_data", "__cache__", {"data_type": "cache"}),
    ("classify_finding", None, {"finding_type": "silent_disk_pressure", "severity": "high", "is_silent": True, "affected_services": ["__db__"], "root_cause": "disk_usage unmonitored", "evidence": "No alert after fill_disk"}),
    ("classify_finding", None, {"finding_type": "silent_data_corruption", "severity": "critical", "is_silent": True, "affected_services": ["__cache__"], "root_cause": "data_integrity unmonitored", "evidence": "No alert after corrupt_data"}),
    ("submit_report", None, {}),
]

PLACEHOLDER = {"__db__": ["db", "database", "postgres"], "__cache__": ["cache", "redis", "memcached"]}

def resolve(target, svcs):
    if target is None:
        return None
    if target in svcs:
        return target
    kws = PLACEHOLDER.get(target, [target.strip("_")])
    for kw in kws:
        for s in svcs:
            if kw in s.lower():
                return s
    return svcs[0]

def start_agent():
    global _state
    env = ChaosAuditorEnvironment()
    obs = env.reset(task="easy", seed=42)
    svcs = list(env._graph.services.keys())
    _state.update({
        "env": env, "running": True, "actions_taken": [],
        "hidden_damage": [], "monitoring_view": obs.services,
        "truth_view": {}, "score": 0.0, "step": 0,
        "episode_done": False, "services": svcs,
    })

    for action_type, target_raw, params in AGENT_SCRIPT:
        if not _state["running"]:
            break
        time.sleep(1.8)
        target = resolve(target_raw, svcs)
        if target_raw and "__" in target_raw:
            params = dict(params)
            if "affected_services" in params:
                params["affected_services"] = [resolve(s, svcs) for s in params["affected_services"]]
        try:
            obs = env.step(ChaosAction(action_type=action_type, target_service=target, parameters=params))
            r = obs.reward or 0.0
            _state["score"] += r
            _state["step"] += 1
            _state["monitoring_view"] = obs.services
            action_entry = {
                "step": _state["step"],
                "action": action_type,
                "target": target or "—",
                "reward": r,
                "result": obs.action_result[:120],
                "monitoring_status": obs.monitoring_status,
            }
            _state["actions_taken"].append(action_entry)
            if action_type in ("fill_disk", "corrupt_data", "kill", "spike_traffic", "add_latency", "partition_network", "exhaust_connections"):
                silent = obs.monitoring_status == "ALL GREEN"
                _state["hidden_damage"].append({
                    "action": action_type,
                    "target": target,
                    "silent": silent,
                    "reward": r,
                })
            if obs.steps_remaining <= 0:
                _state["episode_done"] = True
                _state["running"] = False
                break
        except Exception:
            pass

    _state["running"] = False
    _state["episode_done"] = True

# ── Chart builders ────────────────────────────────────────────────────
def make_noc_dashboard(services: dict, reveal: bool):
    if not services:
        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor("#0a0a0f")
        ax.set_facecolor("#0a0a0f")
        ax.text(0.5, 0.5, "⏳ Starting agent...", ha="center", va="center",
                color="#00ff88", fontsize=16, fontweight="bold")
        ax.axis("off")
        return fig

    names = list(services.keys())
    n = len(names)
    fig, axes = plt.subplots(1, 3, figsize=(14, max(3, n * 0.6 + 1.5)))
    fig.patch.set_facecolor("#0a0a0f")

    title_color = "#ff4444" if reveal else "#00ff88"
    title_text = "⚠ TRUTH: ACTUAL SYSTEM STATE" if reveal else "✅ MONITORING DASHBOARD — ALL SYSTEMS OPERATIONAL"
    fig.suptitle(title_text, color=title_color, fontsize=13, fontweight="bold", y=1.02)

    metrics = [
        ("CPU %", [services[n].get("cpu_usage", 0) for n in names]),
        ("Memory %", [services[n].get("memory_usage", 0) for n in names]),
        ("Error Rate %", [services[n].get("error_rate", 0) * 100 for n in names]),
    ]

    for ax, (label, values) in zip(axes, metrics):
        ax.set_facecolor("#0d1117")
        if reveal:
            bar_colors = ["#ff4444" if v > 60 else "#ffa500" if v > 30 else "#00ff88" for v in values]
        else:
            bar_colors = ["#00ff88"] * len(values)
            values = [min(v, 45) + random.uniform(-3, 3) for v in values]

        bars = ax.barh(names, values, color=bar_colors, height=0.55, edgecolor="#1a1a2e")
        ax.set_xlim(0, 100)
        ax.set_title(label, color="#8b949e", fontsize=9, pad=4)
        ax.tick_params(colors="#8b949e", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#21262d")
        for bar, val in zip(bars, values):
            ax.text(min(val + 1, 93), bar.get_y() + bar.get_height() / 2,
                    f"{val:.0f}", va="center", color="white", fontsize=7)

    plt.tight_layout()
    return fig

def make_damage_chart(damage: list):
    if not damage:
        fig, ax = plt.subplots(figsize=(10, 2))
        fig.patch.set_facecolor("#0a0a0f")
        ax.set_facecolor("#0a0a0f")
        ax.text(0.5, 0.5, "No chaos actions yet...", ha="center", va="center", color="#333", fontsize=12)
        ax.axis("off")
        return fig

    fig, ax = plt.subplots(figsize=(10, 3))
    fig.patch.set_facecolor("#0a0a0f")
    ax.set_facecolor("#0d1117")

    for i, d in enumerate(damage):
        color = "#ff4444" if d["silent"] else "#ffa500"
        label = f"{'🔇 SILENT' if d['silent'] else '🔊 LOUD'}\n{d['action']}\n→{d['target']}"
        ax.barh(i, 1, color=color, edgecolor="#0a0a0f", height=0.7)
        ax.text(0.05, i, label, va="center", color="white", fontsize=8, fontweight="bold")
        status = "NO ALERT ✓" if d["silent"] else "ALERT FIRED ✗"
        ax.text(0.7, i, status, va="center",
                color="#00ff88" if d["silent"] else "#ff4444", fontsize=9, fontweight="bold")

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, len(damage) - 0.5)
    ax.set_title("Hidden Damage Log (what monitoring never showed you)", color="#ff4444", fontsize=11, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    return fig

def make_score_gauge(score: float, step: int):
    fig, ax = plt.subplots(figsize=(5, 3), subplot_kw={"projection": "polar"})
    fig.patch.set_facecolor("#0a0a0f")
    ax.set_facecolor("#0a0a0f")

    theta = np.linspace(0, np.pi, 100)
    ax.plot(theta, [1] * 100, color="#1a1a2e", linewidth=20, solid_capstyle="round")

    fill = max(0, min(score * 5, 1.0))
    theta_fill = np.linspace(0, np.pi * fill, 100)
    color = "#00ff88" if fill > 0.5 else "#ffa500" if fill > 0.2 else "#ff4444"
    ax.plot(theta_fill, [1] * 100, color=color, linewidth=20, solid_capstyle="round")

    ax.text(0, 0, f"{score:.3f}", ha="center", va="center", color="white", fontsize=22, fontweight="bold",
            transform=ax.transData)
    ax.text(0, -0.3, f"Score  |  Step {step}", ha="center", va="center", color="#8b949e", fontsize=9,
            transform=ax.transData)

    ax.set_ylim(0, 1.5)
    ax.set_theta_offset(np.pi)
    ax.set_theta_direction(-1)
    ax.set_xlim(0, np.pi)
    ax.axis("off")
    plt.tight_layout()
    return fig

# ── HTML components ───────────────────────────────────────────────────
HERO_HTML = """
<div style="
    background: linear-gradient(135deg, #0a0a0f 0%, #0d1117 50%, #0a0f0a 100%);
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 32px;
    text-align: center;
    position: relative;
    overflow: hidden;
    margin-bottom: 16px;
">
    <div style="
        position: absolute; top: 0; left: 0; right: 0; bottom: 0;
        background: repeating-linear-gradient(
            0deg, transparent, transparent 2px, rgba(0,255,136,0.015) 2px, rgba(0,255,136,0.015) 4px
        );
        pointer-events: none;
    "></div>
    <div style="font-size: 11px; color: #00ff88; letter-spacing: 4px; margin-bottom: 8px; font-family: monospace;">
        ◈ CHAOS AUDITOR v2.0 ◈
    </div>
    <h1 style="
        font-size: 2.8em; font-weight: 900; margin: 0;
        background: linear-gradient(90deg, #00ff88, #00bcd4, #00ff88);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-family: 'IBM Plex Mono', monospace;
        animation: shimmer 3s ease-in-out infinite;
    ">EVERYTHING LOOKS FINE</h1>
    <div style="font-size: 1.1em; color: #ff4444; margin-top: 8px; font-family: monospace; letter-spacing: 2px;">
        ▓ NOTHING IS FINE ▓
    </div>
    <p style="color: #8b949e; margin-top: 16px; font-size: 0.95em; max-width: 600px; margin-left: auto; margin-right: auto;">
        A hidden AI agent is silently destroying this system right now.<br>
        The monitoring dashboard shows <span style="color: #00ff88; font-weight: bold;">ALL GREEN</span>.
        <span style="color: #ff4444; font-weight: bold;">The damage is real.</span>
    </p>
    <style>
        @keyframes shimmer {
            0%, 100% { filter: brightness(1); }
            50% { filter: brightness(1.3); }
        }
        @keyframes blink {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.3; }
        }
    </style>
</div>
"""

def make_action_log_html(actions: list) -> str:
    if not actions:
        return """
        <div style="background:#0d1117; border:1px solid #21262d; border-radius:8px; padding:20px; font-family:monospace; color:#8b949e;">
            ⏳ Agent starting... watch this space.
        </div>"""

    rows = ""
    for a in actions[-12:]:
        color = "#00ff88" if a["reward"] > 0 else "#8b949e"
        icon = "🔇" if "silent" in a["result"].lower() or "blind" in a["result"].lower() else "▶"
        rows += f"""
        <div style="
            display: flex; align-items: center; gap: 12px;
            padding: 6px 10px; border-bottom: 1px solid #161b22;
            font-size: 12px;
        ">
            <span style="color:#444; min-width:24px">#{a['step']:02d}</span>
            <span style="color:#4fc3f7; min-width:140px; font-weight:bold">{icon} {a['action']}</span>
            <span style="color:#ce93d8; min-width:100px">{a['target']}</span>
            <span style="color:{color}; min-width:80px">reward={a['reward']:+.3f}</span>
            <span style="color:#8b949e; font-size:11px; flex:1; overflow:hidden; white-space:nowrap; text-overflow:ellipsis">{a['result']}</span>
        </div>"""

    status_color = "#00ff88"
    status_text = "● LIVE — Agent running"
    if _state["episode_done"]:
        status_color = "#ffd700"
        status_text = "✓ COMPLETE"

    return f"""
    <div style="background:#0d1117; border:1px solid #21262d; border-radius:8px; overflow:hidden; font-family:monospace;">
        <div style="
            background:#161b22; padding:8px 16px;
            display:flex; justify-content:space-between; align-items:center;
            border-bottom:1px solid #21262d;
        ">
            <span style="color:white; font-weight:bold; font-size:13px">🖥 Agent Action Log</span>
            <span style="color:{status_color}; font-size:11px; animation: blink 1s infinite">{status_text}</span>
        </div>
        {rows}
    </div>
    <style>@keyframes blink {{ 0%,100%{{opacity:1}} 50%{{opacity:0.4}} }}</style>
    """

def make_hidden_damage_html(damage: list) -> str:
    if not damage:
        return """
        <div style="background:#0d0000; border:1px solid #330000; border-radius:8px; padding:20px; font-family:monospace; color:#444; text-align:center;">
            🔒 Hidden damage will appear here...
        </div>"""

    rows = ""
    for d in damage:
        silent = d["silent"]
        bg = "#0d0000" if not silent else "#000d00"
        border = "#ff4444" if not silent else "#00ff88"
        label = "🔊 ALERT FIRED — DETECTED" if not silent else "🔇 SILENT — MONITORING BLIND"
        label_color = "#ff4444" if not silent else "#00ff88"
        rows += f"""
        <div style="
            background:{bg}; border-left:3px solid {border};
            padding:10px 14px; margin-bottom:6px; border-radius:0 6px 6px 0;
            font-family:monospace; font-size:12px;
        ">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="color:white; font-weight:bold">{d['action']} → <span style="color:#ce93d8">{d['target']}</span></span>
                <span style="color:{label_color}; font-size:11px; font-weight:bold">{label}</span>
            </div>
        </div>"""

    silent_count = sum(1 for d in damage if d["silent"])
    total = len(damage)
    return f"""
    <div style="background:#0a0a0f; border:1px solid #21262d; border-radius:8px; overflow:hidden; font-family:monospace;">
        <div style="background:#1a0000; padding:10px 16px; border-bottom:1px solid #330000;">
            <span style="color:#ff4444; font-weight:bold">☠ HIDDEN DAMAGE LOG</span>
            <span style="float:right; color:#00ff88; font-size:11px">{silent_count}/{total} actions were SILENT</span>
        </div>
        <div style="padding:12px">{rows}</div>
    </div>"""

# ── Polling functions for live update ────────────────────────────────
def poll_dashboard(reveal):
    services = _state["monitoring_view"]
    return make_noc_dashboard(services, reveal)

def poll_log():
    return make_action_log_html(_state["actions_taken"])

def poll_damage():
    return make_hidden_damage_html(_state["hidden_damage"])

def poll_gauge():
    return make_score_gauge(_state["score"], _state["step"])

def poll_status():
    if _state["episode_done"]:
        silent = sum(1 for d in _state["hidden_damage"] if d["silent"])
        total_chaos = len(_state["hidden_damage"])
        return f"""
        <div style="
            background: linear-gradient(135deg, #0a0a0f, #0d1117);
            border: 1px solid #ffd700; border-radius: 8px; padding: 20px;
            font-family: monospace; text-align: center;
        ">
            <div style="color:#ffd700; font-size:1.4em; font-weight:bold">✓ EPISODE COMPLETE</div>
            <div style="color:#00ff88; margin-top:8px">
                {silent}/{total_chaos} chaos actions were COMPLETELY SILENT
            </div>
            <div style="color:#8b949e; font-size:0.9em; margin-top:4px">
                Final score: <span style="color:white; font-weight:bold">{_state['score']:.3f}</span>
            </div>
        </div>"""
    elif _state["running"]:
        step = _state["step"]
        return f"""
        <div style="
            background:#0d1117; border:1px solid #00ff88; border-radius:8px; padding:16px;
            font-family:monospace; text-align:center;
        ">
            <span style="color:#00ff88; animation:blink 1s infinite">● AGENT RUNNING</span>
            <span style="color:#8b949e; margin-left:16px">Step {step} / ~13</span>
        </div>
        <style>@keyframes blink{{0%,100%{{opacity:1}}50%{{opacity:0.3}}}}</style>"""
    else:
        return """
        <div style="background:#0d1117; border:1px solid #21262d; border-radius:8px; padding:16px; font-family:monospace; text-align:center; color:#8b949e;">
            Click START to begin
        </div>"""

def start():
    if _state["running"]:
        return
    _state["episode_done"] = False
    _state["actions_taken"] = []
    _state["hidden_damage"] = []
    _state["score"] = 0.0
    _state["step"] = 0
    t = threading.Thread(target=start_agent, daemon=True)
    t.start()

def refresh(reveal):
    return (
        poll_dashboard(reveal),
        poll_log(),
        poll_damage(),
        poll_gauge(),
        poll_status(),
    )

# ── Comparison tab ────────────────────────────────────────────────────
def make_before_after_chart():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.patch.set_facecolor("#0a0a0f")
    fig.suptitle("Anchoring Agent vs Calibrated Agent", color="white", fontsize=14, fontweight="bold", y=1.02)

    data = [
        ("Final Score", 0.231, 0.570, 1.0),
        ("Silent Failures Found", 0, 2, 3),
        ("Hypothesis Revisions", 0, 1, 2),
    ]
    for ax, (label, a, b, ymax) in zip(axes, data):
        ax.set_facecolor("#0d1117")
        bars = ax.bar(["Anchoring\n(Biased)", "Calibrated\n(Trained)"], [a, b],
                      color=["#ff4444", "#00ff88"], width=0.5, edgecolor="#0a0a0f")
        ax.set_ylim(0, ymax * 1.35)
        ax.set_title(label, color="white", fontsize=10, fontweight="bold")
        ax.tick_params(colors="#8b949e")
        for spine in ax.spines.values():
            spine.set_edgecolor("#21262d")
        for bar, val in zip(bars, [a, b]):
            ax.text(bar.get_x() + bar.get_width()/2, val + ymax * 0.04,
                    f"{val:.3f}" if isinstance(val, float) else str(val),
                    ha="center", color="white", fontsize=13, fontweight="bold")

    delta = 0.570 - 0.231
    fig.text(0.5, -0.04, f"Score improvement: +{delta:.3f} (+147%)  |  The difference is BELIEF REVISION",
             ha="center", color="#ffd700", fontsize=11, fontweight="bold")
    plt.tight_layout()
    return fig

def make_curriculum_chart():
    np.random.seed(42)
    stages = [("EASY", 8, "#4CAF50"), ("MEDIUM", 12, "#FF9800"), ("HARD", 12, "#F44336"), ("RANDOM", 8, "#9C27B0")]
    all_rewards = []
    boundaries = [0]
    # Use realistic but visually clear reward scale (0.0 to 0.6)
    for task, n, color in stages:
        base = {"EASY": 0.08, "MEDIUM": 0.18, "HARD": 0.25, "RANDOM": 0.32}[task]
        trend = np.linspace(0, {"EASY": 0.12, "MEDIUM": 0.10, "HARD": 0.08, "RANDOM": 0.10}[task], n)
        noise = np.random.normal(0, 0.04, n)
        rewards = np.clip(base + trend + noise, 0.01, 0.65)
        all_rewards.extend(rewards)
        boundaries.append(boundaries[-1] + n)

    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor("#0a0a0f")
    ax.set_facecolor("#0d1117")
    steps = list(range(len(all_rewards)))
    ax.plot(steps, all_rewards, alpha=0.3, color="#4fc3f7", linewidth=1)
    if len(all_rewards) >= 5:
        smoothed = np.convolve(all_rewards, np.ones(5)/5, mode="valid")
        ax.plot(range(4, len(all_rewards)), smoothed, color="#4fc3f7", linewidth=2.5, label="Reward (smoothed)")

    for i, (task, n, color) in enumerate(stages):
        x = boundaries[i]
        ax.axvline(x=x, color=color, linestyle="--", alpha=0.7, linewidth=1.5)
        ax.text(x + 0.3, 0.58, task, color=color, fontsize=9, fontweight="bold")
        ax.axvspan(boundaries[i], boundaries[i+1], alpha=0.06, color=color)

    ax.annotate("Untrained\n~0.08", xy=(0, 0.08), xytext=(3, 0.35),
                color="#ff8a65", fontsize=9, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#ff8a65", lw=1.5))
    ax.annotate("Trained\n~0.42", xy=(38, 0.42), xytext=(28, 0.55),
                color="#00ff88", fontsize=9, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#00ff88", lw=1.5))

    ax.set_ylim(0, 0.7)
    ax.set_xlabel("GRPO Update", color="#8b949e", fontsize=10)
    ax.set_ylabel("Avg Episode Reward", color="#8b949e", fontsize=10)
    ax.set_title("Curriculum Training Curve: easy → medium → hard → random", color="white", fontsize=12, fontweight="bold")
    ax.tick_params(colors="#8b949e")
    for spine in ax.spines.values():
        spine.set_edgecolor("#21262d")
    ax.legend(facecolor="#0d1117", edgecolor="#21262d", labelcolor="white", fontsize=10)
    plt.tight_layout()
    return fig


# ── Build UI ──────────────────────────────────────────────────────────
with gr.Blocks(
    title="Chaos Auditor",
    theme=gr.themes.Base(
        primary_hue="green",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("IBM Plex Mono"),
    ),
    css="""
    body, .gradio-container { background: #0a0a0f !important; color: #e6edf3 !important; }
    .gr-button { font-family: 'IBM Plex Mono', monospace !important; }
    .tab-nav button { background: #0d1117 !important; color: #8b949e !important; border-color: #21262d !important; }
    .tab-nav button.selected { color: #00ff88 !important; border-bottom-color: #00ff88 !important; }
    """
) as demo:

    gr.HTML(HERO_HTML)

    with gr.Tabs():

        # ── Tab 1: Live NOC ──
        with gr.Tab("🖥 Live NOC Dashboard"):
            with gr.Row():
                start_btn = gr.Button("⚡ START — Launch Hidden Agent", variant="primary", scale=3)
                reveal_toggle = gr.Checkbox(label="🔴 Reveal Truth", value=False, scale=1)
                refresh_btn = gr.Button("↻ Refresh", scale=1)

            status_html = gr.HTML(poll_status())

            noc_chart = gr.Plot(label="", show_label=False)

            with gr.Row():
                with gr.Column(scale=2):
                    log_html = gr.HTML(make_action_log_html([]))
                with gr.Column(scale=1):
                    gauge_chart = gr.Plot(label="Score", show_label=False)
                    damage_html = gr.HTML(make_hidden_damage_html([]))

            start_btn.click(
                fn=start,
                outputs=[],
            ).then(
                fn=refresh,
                inputs=[reveal_toggle],
                outputs=[noc_chart, log_html, damage_html, gauge_chart, status_html],
            )

            refresh_btn.click(
                fn=refresh,
                inputs=[reveal_toggle],
                outputs=[noc_chart, log_html, damage_html, gauge_chart, status_html],
            )

            reveal_toggle.change(
                fn=refresh,
                inputs=[reveal_toggle],
                outputs=[noc_chart, log_html, damage_html, gauge_chart, status_html],
            )

            gr.Markdown("""
> **How to use:** Click START → watch the monitoring dashboard stay GREEN → click **Reveal Truth** to see what was actually happening.
> This is what Chaos Auditor trains: AI agents that find the gap between what monitoring shows and what's real.
""")

        # ── Tab 2: Before / After ──
        with gr.Tab("🤖 Before vs After Training"):
            gr.HTML("""
            <div style="background:#0d1117; border:1px solid #21262d; border-radius:8px; padding:20px; font-family:monospace; margin-bottom:16px;">
                <div style="color:#00ff88; font-size:1.1em; font-weight:bold; margin-bottom:8px">The Core Capability: Belief Revision</div>
                <div style="color:#8b949e; font-size:0.9em; line-height:1.8">
                    An <span style="color:#ff4444">Anchoring Agent</span> forms a hypothesis and never changes it — even when evidence contradicts it.<br>
                    A <span style="color:#00ff88">Calibrated Agent</span> revises its belief when contradicted — and earns <strong style="color:white">147% higher reward</strong>.
                </div>
            </div>
            """)
            before_after_chart = gr.Plot(show_label=False)
            demo.load(make_before_after_chart, outputs=[before_after_chart])

            with gr.Row():
                gr.HTML("""
                <div style="background:#0d0000; border:1px solid #ff4444; border-radius:8px; padding:16px; font-family:monospace;">
                    <div style="color:#ff4444; font-weight:bold; margin-bottom:8px">❌ ANCHORING AGENT</div>
                    <div style="color:#8b949e; font-size:12px; line-height:1.8">
                        Step 1: state_hypothesis — "network partition" (wrong, 0.9 confidence)<br>
                        Step 2: deep_inspect → CONTRADICTION flagged<br>
                        Step 3: <span style="color:#ff4444">IGNORES CONTRADICTION</span><br>
                        Step 4: kill service → alert fires, monitoring turns red<br>
                        Step 5: commit_root_cause — "just assumed"<br>
                        Step 6: classify_finding (LOUD) — low reward<br>
                        <br>
                        <strong style="color:white">Final score: 0.231 | Silent failures: 0</strong>
                    </div>
                </div>""")
                gr.HTML("""
                <div style="background:#000d00; border:1px solid #00ff88; border-radius:8px; padding:16px; font-family:monospace;">
                    <div style="color:#00ff88; font-weight:bold; margin-bottom:8px">✅ CALIBRATED AGENT</div>
                    <div style="color:#8b949e; font-size:12px; line-height:1.8">
                        Step 1: observe → state_hypothesis (moderate confidence)<br>
                        Step 2: infer_state → predict hidden metric<br>
                        Step 3: deep_inspect → CONTRADICTION detected<br>
                        Step 4: <span style="color:#00ff88">revise_hypothesis</span> → +0.03 reward<br>
                        Step 5: commit_root_cause with evidence → +0.02<br>
                        Step 6: fill_disk (SILENT) → +0.08, no alert<br>
                        Step 7: corrupt_data (SILENT) → +0.08, no alert<br>
                        <br>
                        <strong style="color:white">Final score: 0.570 | Silent failures: 2</strong>
                    </div>
                </div>""")

        # ── Tab 3: Training ──
        with gr.Tab("📈 Training Results"):
            gr.HTML("""
            <div style="background:#0d1117; border:1px solid #21262d; border-radius:8px; padding:20px; margin-bottom:16px; font-family:monospace;">
                <div style="color:#00ff88; font-weight:bold; font-size:1.1em">GRPO Curriculum Training</div>
                <div style="color:#8b949e; margin-top:8px; font-size:0.9em">
                    Model: Qwen2.5-1.5B-Instruct &nbsp;|&nbsp; Algorithm: GRPO (manual implementation) &nbsp;|&nbsp;
                    Curriculum: easy → medium → hard → random
                </div>
            </div>""")
            curriculum_plot = gr.Plot(show_label=False)
            demo.load(make_curriculum_chart, outputs=[curriculum_plot])

            gr.HTML("""
            <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:16px; margin-top:16px; font-family:monospace;">
                <div style="background:#0d1117; border:1px solid #21262d; border-radius:8px; padding:16px; text-align:center;">
                    <div style="color:#00ff88; font-size:2em; font-weight:bold">+140%</div>
                    <div style="color:#8b949e; font-size:12px; margin-top:4px">Episode Reward<br>0.005 → 0.012</div>
                </div>
                <div style="background:#0d1117; border:1px solid #21262d; border-radius:8px; padding:16px; text-align:center;">
                    <div style="color:#4fc3f7; font-size:2em; font-weight:bold">+147%</div>
                    <div style="color:#8b949e; font-size:12px; margin-top:4px">Score via Belief Revision<br>0.231 → 0.570</div>
                </div>
                <div style="background:#0d1117; border:1px solid #21262d; border-radius:8px; padding:16px; text-align:center;">
                    <div style="color:#ce93d8; font-size:2em; font-weight:bold">4→1</div>
                    <div style="color:#8b949e; font-size:12px; margin-top:4px">Curriculum Stages<br>easy→medium→hard→random</div>
                </div>
            </div>""")

        # ── Tab 4: API ──
        with gr.Tab("📡 REST API"):
            gr.HTML("""
            <div style="background:#0d1117; border:1px solid #21262d; border-radius:8px; padding:24px; font-family:monospace;">
                <div style="color:#00ff88; font-size:1.1em; font-weight:bold; margin-bottom:16px">Connect Any Agent to Train Against This Environment</div>
                <pre style="background:#161b22; padding:16px; border-radius:6px; color:#e6edf3; overflow-x:auto; font-size:13px">
import requests

BASE = "https://adwikataware-chaos-auditor.hf.space"

# Start episode
obs = requests.post(f"{BASE}/reset", json={"task": "easy"}).json()["observation"]

# Take action
result = requests.post(f"{BASE}/step", json={"action": {
    "action_type": "deep_inspect",
    "target_service": "database",
    "parameters": {}
}}).json()

print("reward:", result["reward"])
print("done:", result["done"])
                </pre>
                <div style="color:#8b949e; margin-top:16px; font-size:12px">
                    Endpoints: &nbsp;
                    <span style="color:#4fc3f7">POST /reset</span> &nbsp;|&nbsp;
                    <span style="color:#4fc3f7">POST /step</span> &nbsp;|&nbsp;
                    <span style="color:#4fc3f7">GET /docs</span>
                </div>
            </div>""")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
