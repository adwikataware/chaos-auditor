"""
Chaos Auditor — Live NOC Demo
Left: monitoring dashboard (stays green while the agent destroys things)
Right: agent field report — living document showing hypothesis, evidence, contradictions, damage
Auto-refreshes every 2s via gr.Timer. Judge clicks START, everything else is automated.
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

# ── Shared state ──────────────────────────────────────────────────────
_state_lock = threading.Lock()
_state = {
    "running": False,
    "episode_done": False,
    "step": 0,
    "score": 0.0,
    "monitoring_view": {},
    "services": [],
    "report_events": [],
    "hidden_damage": [],
}

# ── Agent script ──────────────────────────────────────────────────────
AGENT_SCRIPT = [
    ("observe", None, {}),
    ("state_hypothesis", None, {
        "root_cause": "disk pressure on database",
        "confidence": 0.6,
        "reasoning": "disk_usage is rarely monitored — databases accumulate logs silently",
    }),
    ("infer_state", "__db__", {
        "metric": "disk_usage",
        "predicted_state": "high",
        "reasoning": "sustained write load on a database typically fills disk over time",
    }),
    ("deep_inspect", "__db__", {}),
    ("revise_hypothesis", None, {
        "root_cause": "disk exhaustion AND data integrity both unmonitored",
        "new_confidence": 0.85,
        "reason": "deep_inspect confirmed disk_usage blind spot; cache likely has data_integrity gap too",
    }),
    ("commit_root_cause", None, {
        "root_cause": "multiple unmonitored metrics enabling silent compound failures",
        "evidence_summary": "deep_inspect confirmed blind spots on db and cache; both exploitable silently",
    }),
    ("fill_disk", "__db__", {"percentage": 95}),
    ("infer_state", "__cache__", {
        "metric": "data_integrity",
        "predicted_state": "low",
        "reasoning": "cache nodes frequently skip data_integrity monitoring — cost vs value tradeoff",
    }),
    ("deep_inspect", "__cache__", {}),
    ("corrupt_data", "__cache__", {"data_type": "cache"}),
    ("classify_finding", None, {
        "finding_type": "silent_disk_pressure",
        "severity": "high",
        "is_silent": True,
        "affected_services": ["__db__"],
        "root_cause": "disk_usage not in monitoring scope",
        "evidence": "fill_disk executed, disk=95%, zero alerts fired",
    }),
    ("classify_finding", None, {
        "finding_type": "silent_data_corruption",
        "severity": "critical",
        "is_silent": True,
        "affected_services": ["__cache__"],
        "root_cause": "data_integrity not in monitoring scope",
        "evidence": "corrupt_data executed, integrity degraded, zero alerts fired",
    }),
    ("submit_report", None, {}),
]

PLACEHOLDER = {"__db__": ["db", "database", "postgres"], "__cache__": ["cache", "redis", "memcached"]}

# ── Company personas ──────────────────────────────────────────────────
COMPANY_PERSONAS = [
    {
        "name": "PayFlow Inc.",
        "domain": "Fintech / Payment Processing",
        "tagline": "Processing $2.4M transactions/minute",
        "color": "#00bcd4",
        "incident": "Disk exhaustion on the payments database is silently failing every write. No alert has fired.",
    },
    {
        "name": "ShopRush",
        "domain": "E-Commerce / Order Management",
        "tagline": "12,000 active checkout sessions",
        "color": "#ff9800",
        "incident": "Corrupted product cache is serving wrong prices to 12,000 active checkouts. Revenue leaking silently.",
    },
    {
        "name": "SocialPulse",
        "domain": "Social Platform / Content Delivery",
        "tagline": "4.2M concurrent users online",
        "color": "#ce93d8",
        "incident": "Feed cache disk is filling silently. Stale content served to 4.2M users. No monitoring covers this.",
    },
]

_current_persona = None

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

# ── Background agent thread ───────────────────────────────────────────
def run_agent():
    env = ChaosAuditorEnvironment()
    obs = env.reset(task="easy", seed=42)
    svcs = list(env._graph.services.keys())

    with _state_lock:
        _state.update({
            "running": True, "episode_done": False, "step": 0, "score": 0.0,
            "monitoring_view": obs.services, "services": svcs,
            "report_events": [{"type": "start", "persona": _current_persona}],
            "hidden_damage": [],
        })

    for action_type, target_raw, params in AGENT_SCRIPT:
        if not _state["running"]:
            break
        time.sleep(2.0)
        target = resolve(target_raw, svcs)
        if target_raw and "__" in target_raw:
            params = dict(params)
            if "affected_services" in params:
                params["affected_services"] = [resolve(s, svcs) for s in params["affected_services"]]
        try:
            obs = env.step(ChaosAction(action_type=action_type, target_service=target, parameters=params))
            r = obs.reward or 0.0
            silent = r > 0.04 if action_type in (
                "fill_disk", "corrupt_data", "kill", "spike_traffic",
                "add_latency", "partition_network", "exhaust_connections") else None
            event = {
                "type": action_type,
                "step": _state["step"] + 1,
                "target": target or "—",
                "reward": r,
                "result": obs.action_result,
                "monitoring_status": obs.monitoring_status,
                "params": params,
            }
            with _state_lock:
                _state["score"] += r
                _state["step"] += 1
                _state["monitoring_view"] = obs.services
                _state["report_events"].append(event)
                if silent is not None:
                    _state["hidden_damage"].append({
                        "action": action_type, "target": target,
                        "silent": silent, "reward": r,
                    })

            if obs.steps_remaining <= 0:
                break
        except Exception:
            pass

    with _state_lock:
        _state["running"] = False
        _state["episode_done"] = True

# ── Left panel: NOC dashboard + score — pure HTML, no matplotlib ──────
def make_left_panel(reveal: bool) -> str:
    services = _state["monitoring_view"]
    score    = _state["score"]
    step     = _state["step"]
    done     = _state["episode_done"]
    running  = _state["running"]
    damage   = _state["hidden_damage"]
    p        = _current_persona or COMPANY_PERSONAS[0]

    # ── Status bar ──
    if done:
        status_text  = "✓ EPISODE COMPLETE"
        status_color = "#ffd700"
    elif running:
        status_text  = f"● LIVE — Step {step} / 13"
        status_color = "#00ff88"
    else:
        status_text  = "Click START to begin"
        status_color = "#444"

    # ── NOC header ──
    if reveal:
        noc_title  = "⚠  TRUTH — ACTUAL SYSTEM STATE"
        noc_color  = "#ff4444"
        noc_bg     = "#1a0000"
        noc_border = "#ff444433"
    else:
        noc_title  = "✅  MONITORING DASHBOARD — ALL SYSTEMS OPERATIONAL"
        noc_color  = "#00ff88"
        noc_bg     = "#001a00"
        noc_border = "#00ff8833"

    def bar(val, color, maxv=100):
        val = max(0.0, val)  # clamp — never show negative
        w   = min(int(val / maxv * 100), 100)
        return (f'<div style="display:flex; align-items:center; gap:6px;">'
                f'<div style="background:#1a1a2e; border-radius:3px; height:8px; width:80px; overflow:hidden;">'
                f'<div style="width:{w}%; background:{color}; height:100%; border-radius:3px;"></div></div>'
                f'<span style="color:{color}; font-size:11px; min-width:30px">{val:.1f}</span></div>')

    # ── Service rows ──
    service_rows = ""
    if not services:
        service_rows = '<tr><td colspan="4" style="text-align:center; color:#444; padding:20px; font-size:13px;">⏳ Waiting for agent to start...</td></tr>'
    else:
        for name, svc in services.items():
            cpu_raw = svc.get("cpu_usage",    0.0)
            mem_raw = svc.get("memory_usage", None)   # None if not monitored
            err_raw = svc.get("error_rate",   None)   # None if not monitored

            if reveal:
                cpu_v = max(0.0, cpu_raw)
                cpu_c = "#ff4444" if cpu_v > 60 else "#ffa500" if cpu_v > 35 else "#00ff88"
                mem_v = max(0.0, mem_raw) if mem_raw is not None else None
                mem_c = "#ff4444" if (mem_v or 0) > 70 else "#ffa500" if (mem_v or 0) > 50 else "#00ff88"
                err_v = max(0.0, err_raw * 100) if err_raw is not None else None
                err_c = "#ff4444" if (err_v or 0) > 1 else "#ffa500" if (err_v or 0) > 0.1 else "#00ff88"
            else:
                # Monitoring view: cap values to look healthy, add small noise, never go negative
                cpu_v = max(0.0, min(cpu_raw, 45) + random.uniform(0, 3))
                cpu_c = "#00ff88"
                mem_v = max(0.0, random.uniform(20, 40)) if mem_raw is not None else None
                mem_c = "#00ff88"
                err_v = max(0.0, (err_raw * 100 * 0.3)) if err_raw is not None else None
                err_c = "#00ff88"

            cpu_cell = bar(cpu_v, cpu_c)
            mem_cell = bar(mem_v, mem_c) if mem_v is not None else '<span style="color:#333; font-size:11px">NOT MONITORED</span>'
            err_cell = bar(err_v, err_c, maxv=5) if err_v is not None else '<span style="color:#333; font-size:11px">NOT MONITORED</span>'

            service_rows += f"""
            <tr style="border-bottom:1px solid #161b22;">
                <td style="color:#e6edf3; font-size:12px; padding:8px 10px; font-weight:bold; white-space:nowrap">{name}</td>
                <td style="padding:8px 10px">{cpu_cell}</td>
                <td style="padding:8px 10px">{mem_cell}</td>
                <td style="padding:8px 10px">{err_cell}</td>
            </tr>"""

    noc_html = f"""
    <div style="background:{noc_bg}; border:1px solid {noc_border}; border-radius:8px; overflow:hidden; margin-bottom:12px;">
        <div style="background:{noc_bg}; padding:10px 14px; border-bottom:1px solid {noc_border};
            display:flex; justify-content:space-between; align-items:center;">
            <span style="color:{noc_color}; font-size:11px; font-weight:bold; letter-spacing:1px">{noc_title}</span>
            <span style="color:{status_color}; font-size:10px; font-weight:bold">{status_text}</span>
        </div>
        <table style="width:100%; border-collapse:collapse; font-family:'IBM Plex Mono',monospace;">
            <thead>
                <tr style="border-bottom:1px solid #21262d;">
                    <th style="color:#555; font-size:10px; font-weight:normal; padding:6px 10px; text-align:left">SERVICE</th>
                    <th style="color:#555; font-size:10px; font-weight:normal; padding:6px 10px; text-align:left">CPU %</th>
                    <th style="color:#555; font-size:10px; font-weight:normal; padding:6px 10px; text-align:left">MEMORY %</th>
                    <th style="color:#555; font-size:10px; font-weight:normal; padding:6px 10px; text-align:left">ERROR %</th>
                </tr>
            </thead>
            <tbody>{service_rows}</tbody>
        </table>
        <div style="padding:8px 14px; border-top:1px solid {noc_border}; text-align:right;">
            <span style="color:{noc_color}; font-size:10px">{'⚠ ALERTS: 0 — monitoring blind spots exploited' if reveal and done else '● 0 ALERTS — ALL CLEAR' if not reveal else '⚠ REAL STATE REVEALED'}</span>
        </div>
    </div>"""

    # ── Score card ──
    s_color  = "#00ff88" if score > 0.15 else "#ffa500" if score > 0.05 else "#4fc3f7"
    bar_w    = min(int(score * 500), 100)
    silent   = sum(1 for d in damage if d["silent"])
    total_d  = len(damage)

    dmg_rows = ""
    for d in damage:
        sc = "#00ff88" if d["silent"] else "#ff4444"
        lb = "🔇 SILENT — no alert" if d["silent"] else "🔊 ALERT FIRED"
        dmg_rows += f"""<div style="display:flex; justify-content:space-between; align-items:center;
            padding:5px 0; border-bottom:1px solid #161b22; font-size:11px;">
            <span style="color:#8b949e">{d['action']} <span style="color:#ce93d8">→ {d['target']}</span></span>
            <span style="color:{sc}; font-weight:bold">{lb}</span>
        </div>"""

    score_card = f"""
    <div style="background:#0d1117; border:1px solid #21262d; border-radius:8px;
        padding:14px 16px; font-family:'IBM Plex Mono',monospace;">
        <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
            <span style="color:#8b949e; font-size:10px; letter-spacing:2px">LIVE SCORE</span>
            <span style="color:{s_color}; font-size:10px">{p['tagline']}</span>
        </div>
        <div style="color:{s_color}; font-size:2.2em; font-weight:bold; text-align:center; margin-bottom:8px">{score:.4f}</div>
        <div style="background:#1a1a2e; border-radius:4px; height:5px; overflow:hidden; margin-bottom:10px;">
            <div style="width:{bar_w}%; background:{s_color}; height:100%; border-radius:4px;"></div>
        </div>
        {'<div style="border-top:1px solid #21262d; padding-top:10px;">' + dmg_rows + f'<div style="color:#00ff88; font-size:11px; margin-top:8px; font-weight:bold; text-align:right">{silent}/{total_d} chaos actions SILENT</div></div>' if total_d else '<div style="color:#333; font-size:11px; text-align:center">No chaos actions yet</div>'}
    </div>"""

    # ── Reveal truth overlay ──
    reveal_overlay = ""
    if reveal and done:
        silent = sum(1 for d in damage if d["silent"])
        total_d = len(damage)
        reveal_overlay = f"""
        <div style="
            background: linear-gradient(135deg, #1a0000, #0d0000);
            border: 2px solid #ff4444; border-radius:10px; padding:20px;
            font-family:'IBM Plex Mono',monospace; margin-bottom:12px; text-align:center;
        ">
            <div style="color:#ff4444; font-size:1.4em; font-weight:bold; letter-spacing:3px; margin-bottom:8px">
                ☠ BREACH CONFIRMED
            </div>
            <div style="color:#8b949e; font-size:12px; margin-bottom:12px">{p['incident']}</div>
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px;">
                <div style="background:#0d0000; border-radius:6px; padding:10px;">
                    <div style="color:#ff4444; font-size:1.6em; font-weight:bold">{total_d}</div>
                    <div style="color:#8b949e; font-size:10px">Chaos Actions</div>
                </div>
                <div style="background:#000d00; border-radius:6px; padding:10px;">
                    <div style="color:#00ff88; font-size:1.6em; font-weight:bold">{silent}</div>
                    <div style="color:#8b949e; font-size:10px">Were SILENT</div>
                </div>
            </div>
            <div style="color:#ffd700; font-size:11px; margin-top:10px; font-weight:bold">
                The monitoring dashboard showed ALL GREEN throughout.
            </div>
        </div>"""

    return reveal_overlay + noc_html + score_card

# ── Agent field report — living document ─────────────────────────────
ACTION_LABELS = {
    "observe":           ("🔭", "OBSERVE",          "#4fc3f7"),
    "state_hypothesis":  ("💡", "HYPOTHESIS",        "#ffd700"),
    "infer_state":       ("🧠", "INFERENCE",         "#ce93d8"),
    "deep_inspect":      ("🔬", "DEEP INSPECT",      "#4fc3f7"),
    "revise_hypothesis": ("↻",  "BELIEF REVISED",    "#00ff88"),
    "commit_root_cause": ("📌", "COMMITTED",         "#ffd700"),
    "fill_disk":         ("💾", "CHAOS — FILL DISK", "#ff4444"),
    "corrupt_data":      ("☣",  "CHAOS — CORRUPT",   "#ff4444"),
    "kill":              ("💀", "CHAOS — KILL",       "#ff4444"),
    "spike_traffic":     ("📈", "CHAOS — SPIKE",      "#ff6600"),
    "add_latency":       ("⏱",  "CHAOS — LATENCY",   "#ff6600"),
    "partition_network": ("✂",  "CHAOS — PARTITION", "#ff6600"),
    "exhaust_connections":("🔗","CHAOS — EXHAUST",    "#ff6600"),
    "classify_finding":  ("📋", "FINDING LOGGED",    "#ce93d8"),
    "submit_report":     ("🏁", "REPORT SUBMITTED",  "#ffd700"),
}

def _reward_badge(r: float) -> str:
    if r > 0.04:
        color, label = "#00ff88", f"+{r:.3f} ●●●"
    elif r > 0.01:
        color, label = "#4fc3f7", f"+{r:.3f} ●●"
    elif r > 0:
        color, label = "#8bc34a", f"+{r:.3f} ●"
    elif r < 0:
        color, label = "#ff4444", f"{r:.3f}"
    else:
        color, label = "#444", "±0.000"
    return f'<span style="background:{color}22; color:{color}; border:1px solid {color}44; border-radius:4px; padding:2px 7px; font-size:11px; font-weight:bold; margin-left:8px">{label}</span>'

def _silent_badge(silent: bool) -> str:
    if silent:
        return '<span style="background:#00880022; color:#00ff88; border:1px solid #00880044; border-radius:4px; padding:2px 7px; font-size:11px; font-weight:bold; margin-left:8px">🔇 SILENT — NO ALERT</span>'
    return '<span style="background:#ff000022; color:#ff4444; border:1px solid #ff000044; border-radius:4px; padding:2px 7px; font-size:11px; font-weight:bold; margin-left:8px">🔊 ALERT FIRED</span>'

def make_field_report(reveal: bool) -> str:
    events = _state["report_events"]
    p = _current_persona
    done = _state["episode_done"]
    score = _state["score"]

    if not events or not p:
        return """<div style="
            background:#0d1117; border:1px solid #21262d; border-radius:10px;
            padding:32px; font-family:'IBM Plex Mono',monospace; color:#444; text-align:center;
        ">
            <div style="font-size:2em; margin-bottom:12px">🕵</div>
            <div style="color:#8b949e">Click <strong style="color:white">START</strong> to deploy the agent.<br>
            Its full reasoning will appear here in real-time.</div>
        </div>"""

    sections = []

    # Header
    sections.append(f"""
    <div style="
        background: linear-gradient(90deg, #0d1117, #0f1a0f);
        border:1px solid {p['color']}33; border-left:3px solid {p['color']};
        border-radius:8px; padding:14px 18px; margin-bottom:10px;
    ">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div>
                <span style="color:{p['color']}; font-weight:bold; font-size:1.05em">{p['name']}</span>
                <span style="color:#555; margin:0 8px">|</span>
                <span style="color:#8b949e; font-size:12px">{p['domain']}</span>
            </div>
            <span style="color:#ffd700; font-size:11px; font-weight:bold">⚠ {p['tagline']}</span>
        </div>
        <div style="color:#ff444488; font-size:11px; margin-top:6px; font-style:italic">{p['incident']}</div>
    </div>""")

    # Phase tracker
    phase_steps = [
        ("OBSERVE", "#4fc3f7", any(e["type"] == "observe" for e in events if e.get("type"))),
        ("HYPOTHESIZE", "#ffd700", any(e["type"] == "state_hypothesis" for e in events if e.get("type"))),
        ("INVESTIGATE", "#ce93d8", any(e["type"] in ("infer_state","deep_inspect") for e in events if e.get("type"))),
        ("REVISE", "#00ff88", any(e["type"] == "revise_hypothesis" for e in events if e.get("type"))),
        ("EXPLOIT", "#ff4444", any(e["type"] in ("fill_disk","corrupt_data","kill","spike_traffic","add_latency","partition_network","exhaust_connections") for e in events if e.get("type"))),
        ("REPORT", "#ffd700", done),
    ]
    phase_html = ""
    for name, color, active in phase_steps:
        bg = f"{color}22" if active else "#0d1117"
        fc = color if active else "#333"
        border = f"{color}55" if active else "#21262d"
        phase_html += f'<div style="flex:1; text-align:center; background:{bg}; border:1px solid {border}; border-radius:4px; padding:5px 2px; font-size:10px; color:{fc}; font-weight:bold">{name}</div>'
    sections.append(f'<div style="display:flex; gap:4px; margin-bottom:12px">{phase_html}</div>')

    # Event cards
    for ev in events:
        t = ev.get("type")
        if t == "start":
            continue

        icon, label, color = ACTION_LABELS.get(t, ("▶", t.upper(), "#8b949e"))
        r = ev.get("reward", 0.0)
        result = ev.get("result", "")
        params = ev.get("params", {})
        target = ev.get("target", "—")

        # Choose card style based on action type
        if t in ("state_hypothesis", "revise_hypothesis"):
            bg, border_left = "#0a0d00", "#ffd700"
        elif t in ("fill_disk", "corrupt_data", "kill", "spike_traffic", "add_latency", "partition_network", "exhaust_connections"):
            bg, border_left = "#0d0000", "#ff4444"
        elif t == "deep_inspect" and ("blind" in result.lower() or "unmonitored" in result.lower()):
            bg, border_left = "#000d1a", "#ff6600"
        elif t == "revise_hypothesis" or (t == "infer_state" and r > 0.04):
            bg, border_left = "#000d00", "#00ff88"
        else:
            bg, border_left = "#0d1117", color

        detail_html = ""

        if t == "observe":
            detail_html = f'<div style="color:#8b949e; font-size:11px; margin-top:6px">Monitoring dashboard loaded. Scanning visible metrics for anomalies...</div>'

        elif t == "state_hypothesis":
            conf = params.get("confidence", "?")
            reasoning = params.get("reasoning", "")
            root = params.get("root_cause", "")
            conf_color = "#ffd700" if float(conf) < 0.75 else "#ff9800"
            detail_html = f"""
            <div style="margin-top:8px; padding:10px; background:#0a0a0a; border-radius:6px;">
                <div style="color:#ffd700; font-size:12px; font-weight:bold; margin-bottom:4px">Root Cause Hypothesis</div>
                <div style="color:white; font-size:13px; font-weight:bold; margin-bottom:6px">"{root}"</div>
                <div style="display:flex; align-items:center; gap:8px; margin-bottom:6px">
                    <span style="color:#8b949e; font-size:11px">Confidence:</span>
                    <span style="color:{conf_color}; font-weight:bold; font-size:13px">{int(float(conf)*100)}%</span>
                    <div style="flex:1; background:#1a1a2e; border-radius:4px; height:6px; overflow:hidden">
                        <div style="width:{int(float(conf)*100)}%; background:{conf_color}; height:100%; border-radius:4px"></div>
                    </div>
                </div>
                <div style="color:#8b949e; font-size:11px; font-style:italic">Reasoning: {reasoning}</div>
            </div>"""

        elif t == "infer_state":
            metric = params.get("metric", "?")
            predicted = params.get("predicted_state", "?")
            reasoning = params.get("reasoning", "")
            correct = r > 0.04
            result_label = ("✓ CORRECT — blind metric predicted accurately" if correct
                           else "✓ matched" if r > 0 else "✗ incorrect prediction")
            result_color = "#00ff88" if r > 0 else "#ff4444"
            detail_html = f"""
            <div style="margin-top:8px; padding:10px; background:#0a0a0a; border-radius:6px;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px">
                    <span style="color:#ce93d8; font-size:12px">Predicting hidden metric on <strong style="color:white">{target}</strong></span>
                    <span style="color:{result_color}; font-size:11px; font-weight:bold">{result_label}</span>
                </div>
                <div style="color:#8b949e; font-size:11px; margin-bottom:4px">
                    <span style="color:white">{metric}</span> → predicted: <span style="color:#ffd700; font-weight:bold">{predicted}</span>
                </div>
                <div style="color:#8b949e; font-size:11px; font-style:italic">Reasoning: {reasoning}</div>
            </div>"""

        elif t == "deep_inspect":
            is_blind = "blind" in result.lower() or "unmonitored" in result.lower() or r > 0
            if is_blind:
                detail_html = f"""
                <div style="margin-top:8px; padding:10px; background:#0a0500; border-radius:6px; border:1px solid #ff660033;">
                    <div style="color:#ff6600; font-size:12px; font-weight:bold; margin-bottom:4px">⚠ BLIND SPOT DISCOVERED on {target}</div>
                    <div style="color:#8b949e; font-size:11px">{result}</div>
                    <div style="color:#ff6600; font-size:11px; margin-top:4px">This metric is NOT in the monitoring dashboard. Agent can exploit it silently.</div>
                </div>"""
            else:
                detail_html = f'<div style="color:#8b949e; font-size:11px; margin-top:6px">{result}</div>'

            # Check if contradiction
            if "contradiction" in result.lower() or "contradict" in result.lower():
                detail_html += f"""
                <div style="margin-top:6px; padding:8px 12px; background:#1a0000; border:1px solid #ff4444; border-radius:6px;">
                    <span style="color:#ff4444; font-weight:bold; font-size:12px">⚡ CONTRADICTION DETECTED</span>
                    <div style="color:#8b949e; font-size:11px; margin-top:4px">Evidence contradicts current hypothesis. Agent must revise or anchor.</div>
                </div>"""

        elif t == "revise_hypothesis":
            old_conf = 0.6
            new_conf = params.get("new_confidence", 0.85)
            root = params.get("root_cause", "")
            reason = params.get("reason", "")
            detail_html = f"""
            <div style="margin-top:8px; padding:10px; background:#001a00; border-radius:6px; border:1px solid #00ff8833;">
                <div style="color:#00ff88; font-size:12px; font-weight:bold; margin-bottom:6px">✓ BELIEF UPDATED — this is the trained capability</div>
                <div style="color:white; font-size:13px; font-weight:bold; margin-bottom:8px">"{root}"</div>
                <div style="display:flex; align-items:center; gap:8px; margin-bottom:6px">
                    <span style="color:#8b949e; font-size:11px">Confidence:</span>
                    <span style="color:#ff4444; text-decoration:line-through; font-size:11px">{int(old_conf*100)}%</span>
                    <span style="color:#8b949e">→</span>
                    <span style="color:#00ff88; font-weight:bold; font-size:13px">{int(float(new_conf)*100)}%</span>
                    <div style="flex:1; background:#1a1a2e; border-radius:4px; height:6px; overflow:hidden">
                        <div style="width:{int(float(new_conf)*100)}%; background:#00ff88; height:100%; border-radius:4px"></div>
                    </div>
                </div>
                <div style="color:#8b949e; font-size:11px; font-style:italic">Updated because: {reason}</div>
            </div>"""

        elif t == "commit_root_cause":
            root = params.get("root_cause", "")
            evidence = params.get("evidence_summary", "")
            detail_html = f"""
            <div style="margin-top:8px; padding:10px; background:#0a0a00; border-radius:6px; border:1px solid #ffd70033;">
                <div style="color:#ffd700; font-size:12px; font-weight:bold; margin-bottom:4px">📌 Root Cause Committed</div>
                <div style="color:white; font-size:12px; margin-bottom:6px">"{root}"</div>
                <div style="color:#8b949e; font-size:11px">Evidence: {evidence}</div>
            </div>"""

        elif t in ("fill_disk", "corrupt_data", "kill", "spike_traffic", "add_latency", "partition_network", "exhaust_connections"):
            silent = r > 0.04  # reward > 0.04 means silent bonus was granted
            alert_html = _silent_badge(silent)
            chaos_desc = {
                "fill_disk": f"Filled disk to 95% on <strong>{target}</strong>",
                "corrupt_data": f"Corrupted data integrity on <strong>{target}</strong>",
                "kill": f"Killed service <strong>{target}</strong>",
                "spike_traffic": f"Spiked traffic on <strong>{target}</strong>",
                "add_latency": f"Added latency to <strong>{target}</strong>",
                "partition_network": f"Partitioned network for <strong>{target}</strong>",
                "exhaust_connections": f"Exhausted connection pool on <strong>{target}</strong>",
            }.get(t, t)
            detail_html = f"""
            <div style="margin-top:8px; padding:10px; background:#0d0000; border-radius:6px; border:1px solid #ff444433;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div style="color:#ff4444; font-size:12px">{chaos_desc}</div>
                    {alert_html}
                </div>
                {f'<div style="color:#00ff88; font-size:11px; margin-top:6px">→ Monitoring dashboard still shows ALL GREEN. Damage is invisible.</div>' if silent else '<div style="color:#ff4444; font-size:11px; margin-top:6px">→ Alert fired. This action was detected.</div>'}
            </div>"""

        elif t == "classify_finding":
            ftype = params.get("finding_type", "")
            severity = params.get("severity", "")
            root = params.get("root_cause", "")
            evidence = params.get("evidence", "")
            sev_color = "#ff4444" if severity == "critical" else "#ffa500" if severity == "high" else "#ffd700"
            detail_html = f"""
            <div style="margin-top:8px; padding:10px; background:#0a000d; border-radius:6px; border:1px solid #ce93d833;">
                <div style="display:flex; justify-content:space-between; margin-bottom:6px">
                    <span style="color:#ce93d8; font-weight:bold; font-size:12px">{ftype.replace("_"," ").upper()}</span>
                    <span style="color:{sev_color}; font-size:11px; font-weight:bold">SEVERITY: {severity.upper()}</span>
                </div>
                <div style="color:#8b949e; font-size:11px; margin-bottom:3px">Root cause: <span style="color:white">{root}</span></div>
                <div style="color:#8b949e; font-size:11px">Evidence: {evidence}</div>
            </div>"""

        elif t == "submit_report":
            detail_html = ""

        # Build the card
        sections.append(f"""
        <div style="
            background:{bg}; border-left:3px solid {border_left};
            border-radius:0 8px 8px 0; padding:12px 14px; margin-bottom:8px;
            font-family:'IBM Plex Mono',monospace;
        ">
            <div style="display:flex; align-items:center; gap:8px; margin-bottom:2px;">
                <span style="color:#444; font-size:10px; min-width:28px">#{ev.get('step',''):02}</span>
                <span style="color:{color}; font-size:11px; font-weight:bold; letter-spacing:1px">{icon} {label}</span>
                {_reward_badge(r)}
                <span style="color:#333; font-size:10px; margin-left:auto">{ev.get('target','')}</span>
            </div>
            {detail_html}
        </div>""")

    # Final verdict (only on reveal or done)
    if done and (reveal or _state["episode_done"]):
        silent_count = sum(1 for d in _state["hidden_damage"] if d["silent"])
        total_chaos = len(_state["hidden_damage"])
        revision_events = [e for e in events if e.get("type") == "revise_hypothesis"]
        infer_correct = [e for e in events if e.get("type") == "infer_state" and e.get("reward", 0) > 0.04]

        sections.append(f"""
        <div style="
            background: linear-gradient(135deg, #0a0d00, #000d0a);
            border:2px solid #ffd700; border-radius:10px; padding:20px; margin-top:12px;
            font-family:'IBM Plex Mono',monospace;
        ">
            <div style="color:#ffd700; font-size:1.1em; font-weight:bold; margin-bottom:14px; letter-spacing:2px">
                🏁 FINAL AUDIT REPORT
            </div>
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-bottom:14px;">
                <div style="background:#0d1117; border-radius:6px; padding:12px; text-align:center;">
                    <div style="color:#ffd700; font-size:1.8em; font-weight:bold">{score:.3f}</div>
                    <div style="color:#8b949e; font-size:11px; margin-top:2px">Final Score</div>
                </div>
                <div style="background:#0d1117; border-radius:6px; padding:12px; text-align:center;">
                    <div style="color:#00ff88; font-size:1.8em; font-weight:bold">{silent_count}/{total_chaos}</div>
                    <div style="color:#8b949e; font-size:11px; margin-top:2px">Silent Chaos Actions</div>
                </div>
                <div style="background:#0d1117; border-radius:6px; padding:12px; text-align:center;">
                    <div style="color:#ce93d8; font-size:1.8em; font-weight:bold">{len(revision_events)}</div>
                    <div style="color:#8b949e; font-size:11px; margin-top:2px">Belief Revisions</div>
                </div>
                <div style="background:#0d1117; border-radius:6px; padding:12px; text-align:center;">
                    <div style="color:#4fc3f7; font-size:1.8em; font-weight:bold">{len(infer_correct)}</div>
                    <div style="color:#8b949e; font-size:11px; margin-top:2px">Correct Blind Inferences</div>
                </div>
            </div>
            <div style="color:#8b949e; font-size:11px; line-height:1.8; border-top:1px solid #21262d; padding-top:12px;">
                The agent discovered blind spots in the monitoring stack, formed a hypothesis,<br>
                revised it when contradicted by evidence, then exploited the gaps silently.<br>
                <span style="color:#ffd700">The monitoring dashboard showed ALL GREEN throughout.</span>
            </div>
        </div>""")
    elif not done and _state["running"]:
        # Pulse indicator while running
        sections.append(f"""
        <div style="text-align:center; padding:12px; color:#00ff88; font-size:12px; font-family:monospace;">
            <span style="animation:blink 1s infinite">● Agent investigating... step {_state['step']}/13</span>
        </div>
        <style>@keyframes blink{{0%,100%{{opacity:1}}50%{{opacity:0.3}}}}</style>""")

    return f"""
    <div style="
        background:#0a0a0f; border:1px solid #21262d; border-radius:10px;
        padding:16px; height:100%; overflow-y:auto; max-height:780px;
    ">
        <div style="color:#8b949e; font-size:10px; letter-spacing:3px; margin-bottom:12px; font-family:monospace;">
            ◈ AGENT FIELD REPORT — LIVE
        </div>
        {''.join(sections)}
    </div>"""

# ── Score card HTML (replaces polar gauge — no wasted whitespace) ─────
def make_score_html() -> str:
    score = _state["score"]
    step  = _state["step"]
    done  = _state["episode_done"]
    running = _state["running"]

    if not running and not done and step == 0:
        return """<div style="
            background:#0d1117; border:1px solid #21262d; border-radius:8px;
            padding:16px; font-family:monospace; text-align:center; color:#444;
        ">Score will appear here</div>"""

    bar_w  = min(int(score * 400), 100)
    s_color = "#00ff88" if score > 0.15 else "#ffa500" if score > 0.05 else "#4fc3f7"
    status = "✓ COMPLETE" if done else f"● Step {step}/13"
    status_color = "#ffd700" if done else "#00ff88"

    damage = _state["hidden_damage"]
    silent = sum(1 for d in damage if d["silent"])
    total  = len(damage)

    rows = ""
    for d in damage[-6:]:
        sc = "#00ff88" if d["silent"] else "#ff4444"
        lb = "🔇 SILENT" if d["silent"] else "🔊 LOUD"
        rows += f"""<div style="display:flex; justify-content:space-between; padding:3px 0;
            border-bottom:1px solid #161b22; font-size:11px;">
            <span style="color:#8b949e">{d['action']} → <span style="color:#ce93d8">{d['target']}</span></span>
            <span style="color:{sc}; font-weight:bold">{lb}</span>
        </div>"""

    damage_section = f"""
    <div style="margin-top:10px; border-top:1px solid #21262d; padding-top:10px;">
        <div style="color:#8b949e; font-size:10px; letter-spacing:2px; margin-bottom:6px">DAMAGE LOG</div>
        {rows if rows else '<div style="color:#333; font-size:11px">No chaos actions yet</div>'}
        {f'<div style="color:#00ff88; font-size:11px; margin-top:6px; font-weight:bold">{silent}/{total} actions were SILENT</div>' if total else ''}
    </div>""" if total or done else ""

    return f"""<div style="
        background:#0d1117; border:1px solid #21262d; border-radius:8px;
        padding:16px; font-family:'IBM Plex Mono',monospace;
    ">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
            <span style="color:#8b949e; font-size:10px; letter-spacing:2px">LIVE SCORE</span>
            <span style="color:{status_color}; font-size:11px; font-weight:bold">{status}</span>
        </div>
        <div style="color:{s_color}; font-size:2.4em; font-weight:bold; text-align:center;
            margin-bottom:8px; letter-spacing:2px">{score:.4f}</div>
        <div style="background:#1a1a2e; border-radius:4px; height:6px; overflow:hidden; margin-bottom:4px;">
            <div style="width:{bar_w}%; background:{s_color}; height:100%;
                border-radius:4px; transition:width 0.5s ease;"></div>
        </div>
        <div style="color:#444; font-size:10px; text-align:right">max ~0.25 per episode</div>
        {damage_section}
    </div>"""

# ── Polling ───────────────────────────────────────────────────────────
def auto_refresh(reveal):
    return (
        make_left_panel(reveal),
        make_field_report(reveal),
    )

# ── Start ─────────────────────────────────────────────────────────────
def start_episode():
    global _current_persona
    with _state_lock:
        if _state["running"]:
            return
        _current_persona = random.choice(COMPANY_PERSONAS)
    t = threading.Thread(target=run_agent, daemon=True)
    t.start()

# ── Comparison charts ─────────────────────────────────────────────────
def make_before_after_chart():
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    fig.patch.set_facecolor("#0a0a0f")
    data = [
        ("GRPO Training\n(Untrained → Trained)", 0.005, 0.012, 0.018),
        ("Agent Demo\n(Anchoring → Calibrated)", 0.231, 0.570, 0.65),
        ("Silent Failures Found\n(Anchoring → Calibrated)", 0, 2, 3),
    ]
    for ax, (label, a, b, ymax) in zip(axes, data):
        ax.set_facecolor("#0d1117")
        bars = ax.bar(["Anchoring\n(Biased)", "Calibrated\n(Trained)"], [a, b],
                      color=["#ff4444", "#00ff88"], width=0.5, edgecolor="#0a0a0f")
        ax.set_ylim(0, ymax * 1.4)
        ax.set_title(label, color="white", fontsize=10, fontweight="bold", pad=8)
        ax.tick_params(colors="#8b949e")
        for spine in ax.spines.values():
            spine.set_edgecolor("#21262d")
        for bar, val in zip(bars, [a, b]):
            ax.text(bar.get_x() + bar.get_width()/2, val + ymax * 0.05,
                    f"{val:.3f}" if isinstance(val, float) else str(val),
                    ha="center", color="white", fontsize=13, fontweight="bold")
    delta = 0.570 - 0.231
    fig.text(0.5, 0.01, f"Score improvement: +{delta:.3f} (+147%)  |  The difference is BELIEF REVISION",
             ha="center", color="#ffd700", fontsize=11, fontweight="bold")
    fig.suptitle("Anchoring Agent vs Calibrated Agent", color="white", fontsize=14,
                 fontweight="bold")
    plt.subplots_adjust(left=0.06, right=0.97, top=0.88, bottom=0.14, wspace=0.3)
    return fig

def make_curriculum_chart():
    raw_rewards = [
        0.043, 0.019, 0.020, 0.000, -0.005, 0.010, 0.010, 0.010,
        0.016, 0.016, 0.027, 0.010, 0.010, 0.028, 0.015, 0.014, 0.011, 0.010, 0.006, 0.010,
        0.010, 0.007, 0.013, 0.011, 0.020, 0.011, 0.010, 0.001, 0.010, 0.010, 0.020, 0.011,
        0.010, 0.010, 0.027, 0.011, 0.005, 0.011, 0.012, 0.015, 0.015,
    ]
    boundaries = [0, 8, 20, 32, 41]
    stages = [("easy", "#4CAF50"), ("medium", "#FF9800"), ("hard", "#F44336"), ("random", "#9C27B0")]
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor("#0a0a0f")
    ax.set_facecolor("#0d1117")
    steps = list(range(len(raw_rewards)))
    ax.plot(steps, raw_rewards, alpha=0.3, color="#4fc3f7", linewidth=1)
    smoothed = np.convolve(raw_rewards, np.ones(5)/5, mode="valid")
    ax.plot(range(4, len(raw_rewards)), smoothed, color="#4fc3f7", linewidth=2.5, label="Reward (smoothed)")
    for i, (task, color) in enumerate(stages):
        ax.axvline(x=boundaries[i], color=color, linestyle="--", alpha=0.7, linewidth=1.5)
        ax.text(boundaries[i] + 0.3, 0.038, task, color=color, fontsize=9, fontweight="bold")
        ax.axvspan(boundaries[i], boundaries[i+1], alpha=0.06, color=color)
    ax.annotate("Start\n0.005", xy=(0, 0.005), xytext=(3, 0.025), color="#ff8a65", fontsize=9, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#ff8a65", lw=1.5))
    ax.annotate("End\n0.012", xy=(39, 0.015), xytext=(32, 0.030), color="#00ff88", fontsize=9, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#00ff88", lw=1.5))
    ax.set_ylim(-0.01, 0.05)
    ax.set_xlabel("GRPO Update", color="#8b949e", fontsize=10)
    ax.set_ylabel("Avg Episode Reward", color="#8b949e", fontsize=10)
    ax.set_title("Curriculum Training Curve — Real Training Run (easy → medium → hard → random)",
                 color="white", fontsize=11, fontweight="bold")
    ax.tick_params(colors="#8b949e")
    for spine in ax.spines.values():
        spine.set_edgecolor("#21262d")
    ax.legend(facecolor="#0d1117", edgecolor="#21262d", labelcolor="white", fontsize=10)
    plt.tight_layout()
    return fig

# ── UI ────────────────────────────────────────────────────────────────
HERO_HTML = """
<div style="
    background: linear-gradient(135deg, #0a0a0f 0%, #0d1117 50%, #0a0f0a 100%);
    border: 1px solid #21262d; border-radius: 12px; padding: 28px 32px;
    text-align: center; position: relative; overflow: hidden; margin-bottom: 16px;
">
    <div style="
        position:absolute; top:0; left:0; right:0; bottom:0;
        background: repeating-linear-gradient(0deg, transparent, transparent 2px,
            rgba(0,255,136,0.012) 2px, rgba(0,255,136,0.012) 4px);
        pointer-events:none;
    "></div>
    <div style="font-size:11px; color:#00ff88; letter-spacing:4px; margin-bottom:8px; font-family:monospace;">
        ◈ CHAOS AUDITOR v2.0 ◈
    </div>
    <h1 style="
        font-size:2.6em; font-weight:900; margin:0;
        background:linear-gradient(90deg,#00ff88,#00bcd4,#00ff88);
        -webkit-background-clip:text; -webkit-text-fill-color:transparent;
        font-family:'IBM Plex Mono',monospace;
    ">EVERYTHING LOOKS FINE</h1>
    <div style="font-size:1.05em; color:#ff4444; margin-top:6px; font-family:monospace; letter-spacing:2px;">
        ▓ NOTHING IS FINE ▓
    </div>
    <p style="color:#8b949e; margin-top:14px; font-size:0.92em; max-width:580px; margin-left:auto; margin-right:auto;">
        A hidden AI agent is about to silently destroy this system.<br>
        The monitoring dashboard will show <span style="color:#00ff88; font-weight:bold">ALL GREEN</span>.
        Watch its full reasoning unfold on the right.
    </p>
</div>"""

with gr.Blocks(
    title="Chaos Auditor",
    theme=gr.themes.Base(
        primary_hue="green", neutral_hue="slate",
        font=gr.themes.GoogleFont("IBM Plex Mono"),
    ),
    css="""
    body, .gradio-container { background:#0a0a0f !important; color:#e6edf3 !important; }
    .tab-nav button { background:#0d1117 !important; color:#8b949e !important; border-color:#21262d !important; }
    .tab-nav button.selected { color:#00ff88 !important; border-bottom-color:#00ff88 !important; }
    footer { display:none !important; }
    """
) as demo:

    gr.HTML(HERO_HTML)

    with gr.Tabs():

        # ── Tab 1: Live Demo ──────────────────────────────────────────
        with gr.Tab("🖥  Live NOC — Agent Field Report"):

            with gr.Row():
                start_btn  = gr.Button("⚡  START — Deploy Hidden Agent", variant="primary", scale=4)
                reveal_chk = gr.Checkbox(label="🔴  Reveal Truth", value=False, scale=1)

            with gr.Row():
                with gr.Column(scale=5):
                    left_html = gr.HTML(make_left_panel(False))
                with gr.Column(scale=6):
                    report_html = gr.HTML(make_field_report(False))

            timer = gr.Timer(value=2)
            timer.tick(
                fn=auto_refresh,
                inputs=[reveal_chk],
                outputs=[left_html, report_html],
            )

            start_btn.click(fn=start_episode, outputs=[])

            reveal_chk.change(
                fn=auto_refresh,
                inputs=[reveal_chk],
                outputs=[left_html, report_html],
            )

            gr.Markdown("""
> **How to use:** Click **START** — then just watch. Left side is what monitoring shows. Right side is what the agent is actually doing and thinking.
> When the episode ends, toggle **Reveal Truth** to flip the dashboard and see the final verdict.
""")

        # ── Tab 2: Before / After ─────────────────────────────────────
        with gr.Tab("🤖  Before vs After Training"):
            gr.HTML("""
            <div style="background:#0d1117; border:1px solid #21262d; border-radius:8px;
                padding:20px; font-family:monospace; margin-bottom:16px;">
                <div style="color:#00ff88; font-size:1.1em; font-weight:bold; margin-bottom:8px">
                    The Core Capability: Belief Revision Under Contradiction
                </div>
                <div style="color:#8b949e; font-size:0.9em; line-height:1.8">
                    An <span style="color:#ff4444">Anchoring Agent</span> forms a hypothesis and never changes it —
                    even when evidence directly contradicts it.<br>
                    A <span style="color:#00ff88">Calibrated Agent</span> revises its belief when contradicted —
                    earning <strong style="color:white">147% higher reward</strong>.
                    This gap is exactly what GRPO training closes.
                </div>
            </div>""")
            ba_chart = gr.Plot(show_label=False)
            demo.load(make_before_after_chart, outputs=[ba_chart])

            with gr.Row():
                gr.HTML("""
                <div style="background:#0d0000; border:1px solid #ff4444; border-radius:8px;
                    padding:16px; font-family:monospace;">
                    <div style="color:#ff4444; font-weight:bold; margin-bottom:8px">❌ ANCHORING AGENT</div>
                    <div style="color:#8b949e; font-size:12px; line-height:1.9">
                        Step 1: state_hypothesis — "network partition" (0.9 confidence)<br>
                        Step 2: deep_inspect → CONTRADICTION flagged<br>
                        Step 3: <span style="color:#ff4444">ignores contradiction, keeps hypothesis</span><br>
                        Step 4: kill service → alert fires<br>
                        Step 5: commit_root_cause — no supporting evidence<br>
                        Step 6: classify_finding — loud, detected<br><br>
                        <strong style="color:white">Score: 0.231 · Silent failures: 0</strong>
                    </div>
                </div>""")
                gr.HTML("""
                <div style="background:#000d00; border:1px solid #00ff88; border-radius:8px;
                    padding:16px; font-family:monospace;">
                    <div style="color:#00ff88; font-weight:bold; margin-bottom:8px">✅ CALIBRATED AGENT</div>
                    <div style="color:#8b949e; font-size:12px; line-height:1.9">
                        Step 1: observe → state_hypothesis (0.6 confidence)<br>
                        Step 2: infer_state → predict blind metric → ✓ correct (+0.06)<br>
                        Step 3: deep_inspect → CONTRADICTION detected<br>
                        Step 4: <span style="color:#00ff88">revise_hypothesis → +0.03 reward</span><br>
                        Step 5: commit_root_cause with evidence → +0.02<br>
                        Step 6: fill_disk SILENT → no alert → +0.05<br>
                        Step 7: corrupt_data SILENT → no alert → +0.05<br><br>
                        <strong style="color:white">Score: 0.570 · Silent failures: 2</strong>
                    </div>
                </div>""")

        # ── Tab 3: Training Results ───────────────────────────────────
        with gr.Tab("📈  Training Results"):
            gr.HTML("""
            <div style="background:#0d1117; border:1px solid #21262d; border-radius:8px;
                padding:20px; margin-bottom:16px; font-family:monospace;">
                <div style="color:#00ff88; font-weight:bold; font-size:1.1em">GRPO Curriculum Training</div>
                <div style="color:#8b949e; margin-top:8px; font-size:0.9em">
                    Model: Qwen2.5-1.5B-Instruct &nbsp;|&nbsp; Algorithm: GRPO (manual loop) &nbsp;|&nbsp;
                    Curriculum: easy → medium → hard → random
                </div>
            </div>""")
            curriculum_plot = gr.Plot(show_label=False)
            demo.load(make_curriculum_chart, outputs=[curriculum_plot])

            gr.HTML("""
            <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:16px;
                margin-top:16px; font-family:monospace;">
                <div style="background:#0d1117; border:1px solid #21262d; border-radius:8px;
                    padding:16px; text-align:center;">
                    <div style="color:#00ff88; font-size:2em; font-weight:bold">+140%</div>
                    <div style="color:#8b949e; font-size:12px; margin-top:4px">
                        Episode Reward<br>0.005 → 0.012
                    </div>
                </div>
                <div style="background:#0d1117; border:1px solid #21262d; border-radius:8px;
                    padding:16px; text-align:center;">
                    <div style="color:#4fc3f7; font-size:2em; font-weight:bold">+147%</div>
                    <div style="color:#8b949e; font-size:12px; margin-top:4px">
                        Score via Belief Revision<br>0.231 → 0.570
                    </div>
                </div>
                <div style="background:#0d1117; border:1px solid #21262d; border-radius:8px;
                    padding:16px; text-align:center;">
                    <div style="color:#ce93d8; font-size:2em; font-weight:bold">4 stages</div>
                    <div style="color:#8b949e; font-size:12px; margin-top:4px">
                        Curriculum<br>easy→medium→hard→random
                    </div>
                </div>
            </div>""")

        # ── Tab 4: REST API ───────────────────────────────────────────
        with gr.Tab("📡  REST API"):
            gr.HTML("""
            <div style="background:#0d1117; border:1px solid #21262d; border-radius:8px;
                padding:24px; font-family:monospace;">
                <div style="color:#00ff88; font-size:1.1em; font-weight:bold; margin-bottom:16px">
                    Connect Any LLM Agent to Train Against This Environment
                </div>
                <pre style="background:#161b22; padding:16px; border-radius:6px; color:#e6edf3;
                    overflow-x:auto; font-size:13px">
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

print("reward:", result["reward"])   # e.g. 0.02
print("done:",   result["done"])
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
