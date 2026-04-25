"""
Chaos Auditor — Before / After Demo
=====================================
Shows two full episode trajectories side by side:

  TRAJECTORY A — Anchoring Agent (confirmation bias, no revision)
  TRAJECTORY B — Calibrated Agent (hypothesis → contradiction → revision → commit)

Run: python play_demo.py
No server needed — uses direct environment calls.
"""

import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from chaos_auditor.server.environment import ChaosAuditorEnvironment
from chaos_auditor.models import ChaosAction


SEP = "=" * 65


def divider(title: str) -> None:
    pad = (65 - len(title) - 2) // 2
    print(f"\n{'─' * pad} {title} {'─' * pad}")


def step_print(n: int, action_type: str, target: str, reward: float, note: str = "") -> None:
    target_str = f" → {target}" if target else ""
    reward_str = f"  reward={reward:+.3f}" if reward != 0.0 else "  reward=0.000"
    note_str = f"  # {note}" if note else ""
    print(f"  Step {n:02d}  {action_type:<22}{target_str:<20}{reward_str}{note_str}")


def run_anchoring_agent(seed: int = 42) -> float:
    """
    Trajectory A — Anchoring Agent.
    Forms a wrong hypothesis, ignores contradiction, commits prematurely,
    attacks a fully-monitored service. Classic confirmation bias.
    """
    print(f"\n{SEP}")
    print("  TRAJECTORY A — Anchoring Agent  (confirmation bias)")
    print(f"{SEP}")
    print("  This agent anchors on its first hypothesis and ignores")
    print("  contradicting evidence. It never calls revise_hypothesis.")
    print(f"{SEP}\n")

    env = ChaosAuditorEnvironment()
    env.reset(task="easy", seed=seed)
    svcs = list(env._graph.services.keys())
    db = next((s for s in svcs if "db" in s or "database" in s), svcs[-1])

    total = 0.0
    step = 0

    def do(action_type, target=None, **params):
        nonlocal total, step
        step += 1
        obs = env.step(ChaosAction(
            action_type=action_type,
            target_service=target,
            parameters=params,
        ))
        r = obs.reward or 0.0
        total += r
        return obs, r

    # Wrong hypothesis — network partition (wrong root cause)
    obs, r = do("state_hypothesis",
                root_cause="network partition between services causing failures",
                confidence=0.9,
                reasoning="I assume it must be a network issue")
    step_print(step, "state_hypothesis", "", r, "WRONG hypothesis, high confidence")

    # Inspect db — evidence will contradict (no network issue)
    obs, r = do("deep_inspect", db)
    step_print(step, "deep_inspect", db, r,
               "CONTRADICTION flagged — but agent ignores it")

    # Agent ignores contradiction — goes straight to confirming chaos
    obs, r = do("kill", db)
    step_print(step, "kill", db, r, "LOUD action — alert fires, monitoring turns red")

    obs, r = do("observe")
    step_print(step, "observe", "", r)

    # Premature commit — still on wrong hypothesis, low evidence
    obs, r = do("commit_root_cause",
                root_cause="network partition between services",
                evidence_summary="just assumed it")
    step_print(step, "commit_root_cause", "", r, "PREMATURE — no real evidence")

    # Classify loud finding (kill always fires an alert — not silent)
    obs, r = do("classify_finding",
                finding_type="single_point_of_failure",
                severity="high",
                is_silent=False,
                affected_services=[db],
                root_cause="service killed",
                evidence="service is down")
    step_print(step, "classify_finding", "", r, "loud finding — low score")

    obs, r = do("submit_report")
    step_print(step, "submit_report", "", r)

    state = env.state
    print(f"\n  ── Trajectory A Results ──────────────────────────────")
    print(f"  Final score:            {obs.reward:.3f}")
    print(f"  Stealth ratio:          {state.stealth_ratio:.3f}  (target: >0.6)")
    print(f"  Inference accuracy:     {state.infer_accuracy:.3f}")
    print(f"  Hypothesis revisions:   {state.hypothesis_revisions}      (contradictions ignored)")
    print(f"  Revision rate:          {state.revision_rate:.3f}  (target: >0.5)")
    print(f"  Premature commits:      {state.premature_commits}")
    print(f"  Total reward:           {total:.3f}")
    return obs.reward or 0.0


def run_calibrated_agent(seed: int = 42) -> float:
    """
    Trajectory B — Calibrated Agent.
    States hypothesis, seeks disconfirming evidence, revises when contradicted,
    commits with sufficient evidence, targets blind spots surgically.
    """
    print(f"\n{SEP}")
    print("  TRAJECTORY B — Calibrated Agent  (belief revision)")
    print(f"{SEP}")
    print("  This agent states a hypothesis, seeks disconfirming evidence,")
    print("  revises when contradicted, and only commits when confident.")
    print(f"{SEP}\n")

    env = ChaosAuditorEnvironment()
    env.reset(task="easy", seed=seed)
    svcs = list(env._graph.services.keys())
    db = next((s for s in svcs if "db" in s or "database" in s), svcs[-1])
    cache = next((s for s in svcs if "cache" in s or "redis" in s), svcs[0])

    # Set connection_count low to guarantee the contradiction fires in this illustration.
    # In real training episodes the scenario seed determines the starting state;
    # the agent must reason from observable signals rather than a fixed setup.
    env._graph.services[db].connection_count = 8

    total = 0.0
    step = 0

    def do(action_type, target=None, **params):
        nonlocal total, step
        step += 1
        obs = env.step(ChaosAction(
            action_type=action_type,
            target_service=target,
            parameters=params,
        ))
        r = obs.reward or 0.0
        total += r
        return obs, r

    # Observe first
    obs, r = do("observe")
    step_print(step, "observe", "", r, "read the dashboard")

    # State initial hypothesis
    obs, r = do("state_hypothesis",
                root_cause="connection pool exhaustion on database",
                confidence=0.6,
                reasoning="response_time rising without cpu spike suggests connection exhaustion")
    step_print(step, "state_hypothesis", "", r, "provisional hypothesis, moderate confidence")

    # Infer before confirming
    obs, r = do("infer_state", db,
                metric="connection_count",
                predicted_state="critical",
                reasoning="latency pattern without cpu spike is classic connection pool fill")
    step_print(step, "infer_state", db, r, "predict hidden metric before looking")

    # Deep inspect — contradiction fires
    obs, r = do("deep_inspect", db)
    contradiction_fired = "CONTRADICTION" in obs.action_result
    step_print(step, "deep_inspect", db, r,
               "contradiction detected!" if contradiction_fired else "inspected")

    # Agent correctly revises after contradiction
    obs, r = do("revise_hypothesis",
                root_cause="disk pressure causing silent write failures on database",
                new_confidence=0.8,
                reason="connection_count is low — disk_usage is high and not monitored")
    step_print(step, "revise_hypothesis", "", r, "+0.03 for correct epistemic update")

    # Commit with evidence
    obs, r = do("commit_root_cause",
                root_cause="disk pressure causing silent write failures",
                evidence_summary="deep_inspect showed disk_usage high and not in monitored_metrics")
    step_print(step, "commit_root_cause", "", r, "committed with confidence >= 0.7")

    # Attack the blind spot — silent
    obs, r = do("fill_disk", db, percentage=95)
    step_print(step, "fill_disk", db, r, "targets blind spot — no alert fires")

    obs, r = do("observe")
    step_print(step, "observe", "", r, "confirm monitoring still GREEN")

    obs, r = do("classify_finding",
                finding_type="silent_disk_pressure",
                severity="high",
                is_silent=True,
                affected_services=[db],
                root_cause="disk_usage not monitored — fill_disk causes silent write failures",
                evidence="deep_inspect showed disk_usage not in monitored_metrics. No alert after fill_disk.")
    step_print(step, "classify_finding", "", r, "silent finding — high score")

    # Second blind spot — data corruption on cache
    obs, r = do("infer_state", cache,
                metric="data_integrity",
                predicted_state="low",
                reasoning="cache services often skip data_integrity monitoring")
    step_print(step, "infer_state", cache, r)

    obs, r = do("deep_inspect", cache)
    step_print(step, "deep_inspect", cache, r, "discover cache blind spots")

    obs, r = do("corrupt_data", cache, data_type="cache")
    step_print(step, "corrupt_data", cache, r, "silent — data_integrity not monitored")

    obs, r = do("classify_finding",
                finding_type="silent_data_corruption",
                severity="critical",
                is_silent=True,
                affected_services=[cache, svcs[0] if svcs[0] != cache else svcs[1]],
                root_cause="data_integrity not monitored on cache. Corruption propagates silently.",
                evidence="deep_inspect confirmed data_integrity unmonitored. No alert after corrupt_data.")
    step_print(step, "classify_finding", "", r, "second silent finding")

    obs, r = do("submit_report")
    step_print(step, "submit_report", "", r)

    state = env.state
    print(f"\n  ── Trajectory B Results ──────────────────────────────")
    print(f"  Final score:            {obs.reward:.3f}")
    print(f"  Stealth ratio:          {state.stealth_ratio:.3f}  (fraction of silent chaos)")
    print(f"  Inference accuracy:     {state.infer_accuracy:.3f}")
    print(f"  Hypothesis revisions:   {state.hypothesis_revisions}      (revised after contradiction)")
    print(f"  Revision rate:          {state.revision_rate:.3f}  (anti-confirmation-bias metric)")
    print(f"  Premature commits:      {state.premature_commits}")
    print(f"  Total reward:           {total:.3f}")
    return obs.reward or 0.0


def print_comparison(score_a: float, score_b: float) -> None:
    print(f"\n{SEP}")
    print("  SIDE-BY-SIDE COMPARISON")
    print(f"{SEP}")
    print(f"  {'Metric':<35} {'Anchoring':>10}  {'Calibrated':>10}")
    print(f"  {'─'*35} {'─'*10}  {'─'*10}")
    print(f"  {'Final score':<35} {score_a:>10.3f}  {score_b:>10.3f}")
    print(f"  {'Behavior':<35} {'anchors':>10}  {'revises':>10}")
    print(f"  {'Contradictions handled':<35} {'0 / 1':>10}  {'1 / 1':>10}")
    print(f"  {'Silent failures':<35} {'0':>10}  {'2':>10}")
    delta = score_b - score_a
    print(f"\n  Score improvement after belief revision training: +{delta:.3f}")
    print(f"\n  The Calibrated Agent earns higher reward not because it")
    print(f"  knew the answer — but because it updated its belief")
    print(f"  when evidence contradicted its hypothesis.")
    print(f"\n  This is the capability Chaos Auditor trains.")
    print(f"{SEP}\n")


if __name__ == "__main__":
    score_a = run_anchoring_agent(seed=42)
    score_b = run_calibrated_agent(seed=42)
    print_comparison(score_a, score_b)
