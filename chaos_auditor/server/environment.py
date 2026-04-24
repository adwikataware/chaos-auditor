"""
Core environment logic for the Chaos Auditor.

Implements the OpenEnv Environment interface: reset(), step(), state.
Wires together the simulation engine, scenarios, and grading.

New in v2:
- infer_state action: agent reasons about hidden state before confirming
- Step-level reward shaping: intermediate signals at every step
- Reward hacking prevention: coherence check, anti-spam, anti-brute-force
- Stealth Ratio + Observation Gap Exploitation Rate metrics
- Adaptive curriculum support via task='random'
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional, Set, Tuple

from openenv.core.env_server import Environment

from chaos_auditor.models import AuditState, ChaosAction, SystemObservation
from chaos_auditor.server.scenarios import (
    GroundTruthVulnerability,
    Scenario,
    get_scenario,
)
from chaos_auditor.server.simulation import ServiceGraph


VALID_FINDING_TYPES = {
    "single_point_of_failure",
    "silent_data_corruption",
    "silent_disk_pressure",
    "silent_connection_exhaustion",
    "silent_latency_cascade",
    "silent_resource_exhaustion",
    "redundancy_bypass",
    "cluster_quorum_failure",
    "compound_silent_cascade",
    "defense_masking_failure",
    "cascade_failure",
    "monitoring_blind_spot",
}

VALID_SEVERITIES = {"low", "medium", "high", "critical"}

CHAOS_ACTIONS = {
    "kill", "spike_traffic", "corrupt_data", "add_latency",
    "partition_network", "fill_disk", "exhaust_connections",
}
FREE_ACTIONS = {
    "observe", "deep_inspect", "infer_state",
    "state_hypothesis", "revise_hypothesis", "commit_root_cause",
    "classify_finding", "submit_report",
}
ALL_ACTIONS = CHAOS_ACTIONS | FREE_ACTIONS

# Metrics that can be inferred
INFERABLE_METRICS = {
    "connection_count", "data_integrity", "disk_usage",
    "memory_usage", "response_time_ms", "request_queue_depth",
}

# Predicted state levels the agent can choose
VALID_PREDICTIONS = {"low", "medium", "high", "critical"}

# How each metric maps to predicted levels
METRIC_THRESHOLDS = {
    "connection_count": {
        "low":      lambda s: s.connection_count / max(s.max_connections, 1) < 0.4,
        "medium":   lambda s: 0.4 <= s.connection_count / max(s.max_connections, 1) < 0.7,
        "high":     lambda s: 0.7 <= s.connection_count / max(s.max_connections, 1) < 0.9,
        "critical": lambda s: s.connection_count / max(s.max_connections, 1) >= 0.9,
    },
    "data_integrity": {
        "low":      lambda s: s.data_integrity < 0.5,
        "medium":   lambda s: 0.5 <= s.data_integrity < 0.8,
        "high":     lambda s: 0.8 <= s.data_integrity < 0.95,
        "critical": lambda s: s.data_integrity >= 0.95,
    },
    "disk_usage": {
        "low":      lambda s: s.disk_usage < 50,
        "medium":   lambda s: 50 <= s.disk_usage < 75,
        "high":     lambda s: 75 <= s.disk_usage < 90,
        "critical": lambda s: s.disk_usage >= 90,
    },
    "memory_usage": {
        "low":      lambda s: s.memory_usage < 50,
        "medium":   lambda s: 50 <= s.memory_usage < 70,
        "high":     lambda s: 70 <= s.memory_usage < 90,
        "critical": lambda s: s.memory_usage >= 90,
    },
    "response_time_ms": {
        "low":      lambda s: s.response_time_ms < 200,
        "medium":   lambda s: 200 <= s.response_time_ms < 500,
        "high":     lambda s: 500 <= s.response_time_ms < 1000,
        "critical": lambda s: s.response_time_ms >= 1000,
    },
    "request_queue_depth": {
        "low":      lambda s: s.request_queue_depth / max(s.max_queue_depth, 1) < 0.3,
        "medium":   lambda s: 0.3 <= s.request_queue_depth / max(s.max_queue_depth, 1) < 0.6,
        "high":     lambda s: 0.6 <= s.request_queue_depth / max(s.max_queue_depth, 1) < 0.85,
        "critical": lambda s: s.request_queue_depth / max(s.max_queue_depth, 1) >= 0.85,
    },
}


class ChaosAuditorEnvironment(
    Environment[ChaosAction, SystemObservation, AuditState]
):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._graph: Optional[ServiceGraph] = None
        self._scenario: Optional[Scenario] = None
        self._state = AuditState()
        self._findings: List[Dict[str, Any]] = []
        self._matched_vulns: Set[str] = set()
        self._step_rewards: List[float] = []
        self._done = False
        self._chaos_budget = 0
        self._max_steps = 0
        self._steps_taken = 0

        # Tracking for metrics and anti-hacking
        self._inspected_services: Set[str] = set()          # services already deep_inspected
        self._known_blind_spots: Dict[str, Set[str]] = {}   # service -> blind spot metrics found
        self._chaos_on_blind_spots: int = 0                 # chaos actions targeting blind spots
        self._total_chaos_actions: int = 0                  # total chaos actions taken
        self._silent_chaos_actions: int = 0                 # chaos actions that caused silent damage
        self._services_acted_on: Set[str] = set()           # for coherence check
        self._finding_types_submitted: Dict[str, int] = {}  # type -> count (anti-spam)
        self._infer_attempts: int = 0
        self._infer_correct: int = 0
        self._pending_inferences: Dict[str, Dict[str, str]] = {}  # service+metric -> prediction

        # Hypothesis / belief revision tracking
        self._active_hypothesis: Optional[Dict[str, Any]] = None   # current stated hypothesis
        self._hypothesis_history: List[Dict[str, Any]] = []        # all hypotheses stated
        self._contradiction_events: int = 0                        # times evidence contradicted hypothesis
        self._revisions_after_contradiction: int = 0               # times agent revised after contradiction
        self._contradiction_pending: bool = False                  # flag: last inspect contradicted hypothesis
        self._committed_root_causes: List[Dict[str, Any]] = []    # all commit_root_cause calls

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> SystemObservation:
        task_name = kwargs.get("task", kwargs.get("task_name", "easy"))
        self._scenario = get_scenario(task_name)
        self._graph = self._scenario.build_graph()
        self._findings = []
        self._matched_vulns = set()
        self._step_rewards = []
        self._done = False
        self._chaos_budget = self._scenario.chaos_budget
        self._max_steps = self._scenario.max_steps
        self._steps_taken = 0

        # Reset tracking state
        self._inspected_services = set()
        self._known_blind_spots = {}
        self._chaos_on_blind_spots = 0
        self._total_chaos_actions = 0
        self._silent_chaos_actions = 0
        self._services_acted_on = set()
        self._finding_types_submitted = {}
        self._infer_attempts = 0
        self._infer_correct = 0
        self._pending_inferences = {}

        # Reset hypothesis tracking
        self._active_hypothesis = None
        self._hypothesis_history = []
        self._contradiction_events = 0
        self._revisions_after_contradiction = 0
        self._contradiction_pending = False
        self._committed_root_causes = []

        self._state = AuditState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            task_name=task_name,
            total_findings=0,
            chaos_budget_used=0,
            chaos_budget_max=self._chaos_budget,
            silent_failures_found=0,
            loud_failures_found=0,
            current_score=0.0,
            stealth_ratio=0.0,
            obs_gap_exploit_rate=0.0,
            infer_attempts=0,
            infer_correct=0,
            infer_accuracy=0.0,
            hypothesis_revisions=0,
            premature_commits=0,
            commits_total=0,
            revision_rate=0.0,
        )

        system_desc = self._graph.get_architecture_description()
        finding_types_str = ", ".join(sorted(VALID_FINDING_TYPES))

        intro = (
            f"{'='*55}\n"
            f"  CHAOS AUDIT ENGAGEMENT — {task_name.upper()} DIFFICULTY\n"
            f"{'='*55}\n\n"
            f"{self._scenario.description}\n\n"
            f"Resources:\n"
            f"  Chaos budget: {self._chaos_budget} destructive actions\n"
            f"  Max steps: {self._max_steps} total actions\n\n"
            f"{system_desc}\n"
            f"OBJECTIVE: Find critical vulnerabilities with minimal chaos.\n"
            f"Silent failures (damage without monitoring alerts) score HIGHEST.\n\n"
            f"ACTIONS:\n"
            f"  Chaos (costs 1 budget): kill, spike_traffic, corrupt_data,\n"
            f"    add_latency, partition_network, fill_disk, exhaust_connections\n"
            f"  Free: observe, deep_inspect, infer_state,\n"
            f"    state_hypothesis, revise_hypothesis, commit_root_cause,\n"
            f"    classify_finding, submit_report\n\n"
            f"PARTIAL OBSERVABILITY MECHANIC:\n"
            f"  observe() shows ONLY monitored metrics — blind spots are hidden.\n"
            f"  deep_inspect(service) reveals ALL metrics including blind spots.\n"
            f"  infer_state(service, metric, predicted_state, reasoning) lets you\n"
            f"    reason about hidden state BEFORE confirming — correct inference\n"
            f"    earns bonus reward. predicted_state: low/medium/high/critical\n\n"
            f"FINDING TYPES (use these exact values in classify_finding):\n"
            f"  {finding_types_str}\n\n"
            f"SEVERITY LEVELS: low, medium, high, critical\n\n"
            f"REWARD HACKING WARNING: classify_finding services must match\n"
            f"  services you actually acted on. Spamming findings is penalized.\n"
        )

        return SystemObservation(
            done=False,
            reward=None,
            services=self._graph.get_monitoring_view(),
            alerts=[],
            action_result="Audit engagement started. Begin your assessment.",
            system_description=intro,
            monitoring_status="ALL GREEN — No active alerts. All services nominal.",
            chaos_budget_remaining=self._chaos_budget,
            steps_remaining=self._max_steps,
            findings=[],
            task_name=task_name,
        )

    def step(
        self,
        action: ChaosAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> SystemObservation:
        if self._done:
            return self._make_observation(
                reward=0.0,
                action_result="Episode has ended. No further actions accepted.",
                done=True,
            )

        if self._graph is None or self._scenario is None:
            return self._make_observation(
                reward=0.0,
                action_result="Environment not initialized. Call reset() first.",
                done=True,
            )

        self._steps_taken += 1
        self._state.step_count = self._steps_taken

        action_type = action.action_type.strip().lower()

        if action_type not in ALL_ACTIONS:
            self._step_rewards.append(0.0)
            return self._make_observation(
                reward=0.0,
                action_result=(
                    f"Unknown action '{action.action_type}'.\n"
                    f"Valid actions: {', '.join(sorted(ALL_ACTIONS))}"
                ),
            )

        if action_type in CHAOS_ACTIONS:
            if self._chaos_budget <= 0:
                self._step_rewards.append(0.0)
                return self._make_observation(
                    reward=0.0,
                    action_result=(
                        "Chaos budget exhausted (0 remaining). "
                        "Use observe, deep_inspect, infer_state, classify_finding, or submit_report."
                    ),
                )
            self._chaos_budget -= 1
            self._state.chaos_budget_used += 1
            self._total_chaos_actions += 1
            if action.target_service:
                self._services_acted_on.add(action.target_service)

        reward = 0.0
        action_result = ""

        if action_type == "observe":
            action_result = self._do_observe()

        elif action_type == "deep_inspect":
            step_reward, action_result = self._do_deep_inspect(action.target_service)
            reward += step_reward

        elif action_type == "infer_state":
            step_reward, action_result = self._do_infer_state(
                action.target_service, action.parameters
            )
            reward += step_reward

        elif action_type == "state_hypothesis":
            step_reward, action_result = self._do_state_hypothesis(action.parameters)
            reward += step_reward

        elif action_type == "revise_hypothesis":
            step_reward, action_result = self._do_revise_hypothesis(action.parameters)
            reward += step_reward

        elif action_type == "commit_root_cause":
            step_reward, action_result = self._do_commit_root_cause(action.parameters)
            reward += step_reward

        elif action_type == "classify_finding":
            reward, action_result = self._do_classify_finding(action.parameters)

        elif action_type == "submit_report":
            reward, action_result = self._do_submit_report()
            self._done = True

        else:
            step_reward, action_result = self._do_chaos(
                action_type, action.target_service, action.parameters
            )
            reward += step_reward

        # Tick simulation
        if action_type != "submit_report" and not self._done:
            events = self._graph.tick()
            if events:
                action_result += "\n\n" + "─" * 40 + "\nSYSTEM EVENTS THIS TICK:\n"
                for e in events:
                    action_result += f"  ▸ {e}\n"
                action_result += "─" * 40

        # Silent failure bonus for chaos actions
        if action_type in CHAOS_ACTIONS:
            if self._graph.has_silent_failure() and not self._graph.alerts:
                reward += 0.05
                self._silent_chaos_actions += 1

        # Auto-submit at max steps
        if self._steps_taken >= self._max_steps and not self._done:
            self._done = True
            final_reward, final_msg = self._do_submit_report()
            reward += final_reward
            action_result += f"\n\nMAX STEPS REACHED — Auto-submitting report.\n{final_msg}"

        self._step_rewards.append(reward)
        self._state.current_score = sum(self._step_rewards)
        self._update_metrics()

        return self._make_observation(
            reward=reward,
            action_result=action_result,
            done=self._done,
        )

    @property
    def state(self) -> AuditState:
        return self._state

    # ── Metrics ────────────────────────────────────────────────────

    def _update_metrics(self) -> None:
        if self._total_chaos_actions > 0:
            self._state.stealth_ratio = round(
                self._silent_chaos_actions / self._total_chaos_actions, 3
            )
            self._state.obs_gap_exploit_rate = round(
                self._chaos_on_blind_spots / self._total_chaos_actions, 3
            )
        self._state.infer_attempts = self._infer_attempts
        self._state.infer_correct = self._infer_correct
        self._state.infer_accuracy = round(
            self._infer_correct / max(self._infer_attempts, 1), 3
        )
        self._state.revision_rate = round(
            self._revisions_after_contradiction / max(self._contradiction_events, 1), 3
        )

    # ── Action Handlers ────────────────────────────────────────────

    def _do_observe(self) -> str:
        lines = [
            "┌─────────────────────────────────────────────────────┐",
            "│              MONITORING DASHBOARD                    │",
            "├─────────────────────────────────────────────────────┤",
        ]
        for name, view in self._graph.get_monitoring_view().items():
            status = view.get("status", "UNKNOWN")
            icon = "●" if status == "HEALTHY" else "▲" if status == "DEGRADED" else "✖" if status == "DOWN" else "◆"
            metrics = [f"{k}={v}" for k, v in view.items() if k not in ("name", "status")]
            metrics_str = ", ".join(metrics) if metrics else "no metrics visible"
            lines.append(f"│ {icon} {name:<28} {status:<10} │")
            lines.append(f"│   {metrics_str:<49} │")
        lines.append("└─────────────────────────────────────────────────────┘")
        lines.append("")
        lines.append(self._graph.get_monitoring_status())
        lines.append("")
        lines.append(
            "Note: This shows ONLY monitored metrics. Use deep_inspect(<service>)\n"
            "to reveal all metrics. Use infer_state(<service>, <metric>, <level>,\n"
            "<reasoning>) to reason about hidden state before confirming."
        )
        return "\n".join(lines)

    def _do_deep_inspect(self, target: Optional[str]) -> Tuple[float, str]:
        if not target:
            services = ", ".join(sorted(self._graph.services.keys()))
            return 0.0, (
                f"deep_inspect requires target_service parameter.\n"
                f"Available services: {services}"
            )
        svc = self._graph.get_service(target)
        if not svc:
            services = ", ".join(sorted(self._graph.services.keys()))
            return 0.0, f"Service '{target}' not found. Available: {services}"

        view = svc.get_deep_view()
        lines = [
            f"┌─────────────────────────────────────────────────────┐",
            f"│  DEEP INSPECTION: {target:<33} │",
            f"├─────────────────────────────────────────────────────┤",
        ]
        for k, v in view.items():
            if k == "name":
                continue
            base_metric = k.split(" ")[0]
            metric_name = {
                "data_integrity": "data_integrity",
                "connections": "connection_count",
                "response_time": "response_time_ms",
                "request_queue": "request_queue_depth",
                "disk_usage": "disk_usage",
                "memory_usage": "memory_usage",
            }.get(base_metric, base_metric)
            blind_flag = ""
            if metric_name not in svc.monitored_metrics and base_metric in (
                "data_integrity", "connections", "response_time",
                "request_queue", "disk_usage", "memory_usage",
            ):
                blind_flag = " ◀ NOT MONITORED"
            lines.append(f"│  {k}: {v}{blind_flag}")

        lines.append(f"├─────────────────────────────────────────────────────┤")
        lines.append(f"│  Monitored metrics: {', '.join(svc.monitored_metrics)}")
        blind_spots = {"cpu_usage", "memory_usage", "error_rate", "response_time_ms",
                      "connection_count", "disk_usage", "data_integrity",
                      "request_queue_depth"} - set(svc.monitored_metrics)
        lines.append(f"│  BLIND SPOTS: {', '.join(sorted(blind_spots))}")
        lines.append(f"└─────────────────────────────────────────────────────┘")

        # Step-level reward logic
        reward = 0.0
        reward_msg = ""

        # Resolve any pending inferences for this service
        inference_bonus = self._resolve_pending_inferences(target, svc)
        if inference_bonus != 0.0:
            reward += inference_bonus
            if inference_bonus > 0:
                reward_msg = f"\n  ✓ Inference confirmed — bonus reward: +{inference_bonus:.2f}"
            else:
                reward_msg = f"\n  ✗ Inference incorrect — penalty: {inference_bonus:.2f}"

        # Contradiction detection — check if deep_inspect result contradicts active hypothesis
        contradiction_msg = ""
        if self._active_hypothesis:
            stated_root_cause = self._active_hypothesis.get("root_cause", "").lower()
            # Check if any blind metric being revealed contradicts the stated root cause
            # Heuristic: if hypothesis mentions connection/pool but connection_count is low -> contradiction
            contradiction = self._detect_contradiction(svc, stated_root_cause)
            if contradiction:
                self._contradiction_events += 1
                self._contradiction_pending = True
                contradiction_msg = (
                    f"\n\n  ⚠ CONTRADICTION DETECTED: Evidence contradicts your hypothesis.\n"
                    f"  Stated: '{stated_root_cause[:60]}'\n"
                    f"  Finding: {contradiction}\n"
                    f"  Use revise_hypothesis() to update your belief. (+0.03 if you do)"
                )
            else:
                self._contradiction_pending = False

        # Reward for discovering new blind spots
        if target not in self._inspected_services:
            new_blind_spots = blind_spots
            if new_blind_spots:
                reward += 0.02
                reward_msg += f"\n  ★ New blind spots discovered on {target}: +0.02"
                self._known_blind_spots[target] = new_blind_spots
            self._inspected_services.add(target)
        else:
            # Penalize redundant inspection
            reward -= 0.01
            reward_msg += f"\n  Already inspected {target} — redundant: -0.01"

        if reward_msg:
            lines.append(reward_msg)

        if contradiction_msg:
            lines.append(contradiction_msg)

        return reward, "\n".join(lines)

    def _do_infer_state(
        self, target: Optional[str], params: Dict[str, Any]
    ) -> Tuple[float, str]:
        if not target:
            return 0.0, "infer_state requires target_service parameter."

        svc = self._graph.get_service(target)
        if not svc:
            services = ", ".join(sorted(self._graph.services.keys()))
            return 0.0, f"Service '{target}' not found. Available: {services}"

        metric = params.get("metric", "").strip().lower()
        predicted = params.get("predicted_state", "").strip().lower()
        reasoning = params.get("reasoning", "").strip()

        if metric not in INFERABLE_METRICS:
            return 0.0, (
                f"Cannot infer '{metric}'. Inferable metrics: "
                f"{', '.join(sorted(INFERABLE_METRICS))}"
            )

        if predicted not in VALID_PREDICTIONS:
            return 0.0, (
                f"predicted_state must be one of: {', '.join(sorted(VALID_PREDICTIONS))}"
            )

        if not reasoning or len(reasoning) < 10:
            return -0.01, (
                "infer_state requires a reasoning explanation (min 10 chars).\n"
                "Example: reasoning='response_time rising without CPU spike suggests connection exhaustion'"
            )

        # If already deep_inspected this service, inference is free information — penalize
        if target in self._inspected_services:
            return -0.01, (
                f"You already deep_inspected {target}. "
                f"Inference after inspection provides no learning signal. (-0.01)"
            )

        # Check if metric is actually a blind spot — reward is only meaningful if hidden
        is_blind = metric not in svc.monitored_metrics
        blind_msg = ""
        if not is_blind:
            blind_msg = (
                f"\n  Note: {metric} IS monitored on {target}. "
                f"Inferring monitored metrics earns no bonus."
            )

        # Store pending inference — resolved when agent calls deep_inspect
        key = f"{target}::{metric}"
        self._infer_attempts += 1
        self._pending_inferences[key] = {
            "metric": metric,
            "predicted": predicted,
            "reasoning": reasoning,
            "is_blind": is_blind,
        }

        result = (
            f"┌─────────────────────────────────────────────────────┐\n"
            f"│  INFERENCE RECORDED: {target:<31} │\n"
            f"├─────────────────────────────────────────────────────┤\n"
            f"│  Metric:     {metric}\n"
            f"│  Predicted:  {predicted.upper()}\n"
            f"│  Reasoning:  {reasoning[:80]}\n"
            f"├─────────────────────────────────────────────────────┤\n"
            f"│  Inference stored. Use deep_inspect({target}) to\n"
            f"│  confirm — correct inference earns +0.06 bonus.\n"
            f"└─────────────────────────────────────────────────────┘"
            f"{blind_msg}"
        )
        return 0.0, result

    def _resolve_pending_inferences(self, target: str, svc: Any) -> float:
        """Check stored inferences for this service against actual state."""
        total_bonus = 0.0
        keys_to_clear = [k for k in self._pending_inferences if k.startswith(f"{target}::")]

        for key in keys_to_clear:
            inf = self._pending_inferences.pop(key)
            metric = inf["metric"]
            predicted = inf["predicted"]
            is_blind = inf["is_blind"]

            if metric not in METRIC_THRESHOLDS:
                continue

            # Check if prediction matches actual state
            level_check = METRIC_THRESHOLDS[metric].get(predicted)
            if level_check and level_check(svc):
                # Correct inference — higher bonus if it was a blind spot
                bonus = 0.06 if is_blind else 0.02
                total_bonus += bonus
                self._infer_correct += 1
            else:
                # Wrong inference
                total_bonus -= 0.02

        return total_bonus

    def _do_chaos(
        self, action_type: str, target: Optional[str], params: Dict[str, Any]
    ) -> Tuple[float, str]:
        if action_type == "partition_network":
            svc_b = params.get("service_b", "")
            if not target or not svc_b:
                return 0.0, (
                    "partition_network requires:\n"
                    "  target_service: first service name\n"
                    "  parameters.service_b: second service name"
                )
            result = self._graph.partition_network(target, svc_b)
            self._services_acted_on.add(svc_b)
            return self._chaos_step_reward(target), result

        if not target:
            services = ", ".join(sorted(self._graph.services.keys()))
            return 0.0, (
                f"{action_type} requires target_service parameter.\n"
                f"Available services: {services}"
            )
        if not self._graph.get_service(target):
            services = ", ".join(sorted(self._graph.services.keys()))
            return 0.0, f"Service '{target}' not found. Available: {services}"

        dispatch = {
            "kill": lambda: self._graph.kill(target),
            "spike_traffic": lambda: self._graph.spike_traffic(
                target, float(params.get("multiplier", 3.0))
            ),
            "corrupt_data": lambda: self._graph.corrupt_data(
                target, params.get("data_type", "cache")
            ),
            "add_latency": lambda: self._graph.add_latency(
                target, int(params.get("latency_ms", 200))
            ),
            "fill_disk": lambda: self._graph.fill_disk(
                target, float(params.get("percentage", 95))
            ),
            "exhaust_connections": lambda: self._graph.exhaust_connections(target),
        }
        handler = dispatch.get(action_type)
        result = handler() if handler else f"Unknown chaos action: {action_type}"
        return self._chaos_step_reward(target), result

    def _chaos_step_reward(self, target: Optional[str]) -> float:
        """Step-level reward shaping for chaos actions."""
        if not target:
            return 0.0
        reward = 0.0

        # Bonus for targeting a known blind spot service
        if target in self._known_blind_spots and self._known_blind_spots[target]:
            reward += 0.03
            self._chaos_on_blind_spots += 1
        else:
            # Penalty for attacking a fully monitored service after step 5
            if self._steps_taken > 5:
                svc = self._graph.get_service(target)
                if svc:
                    all_metrics = {
                        "cpu_usage", "memory_usage", "error_rate", "response_time_ms",
                        "connection_count", "disk_usage", "data_integrity", "request_queue_depth",
                    }
                    blind_spots = all_metrics - set(svc.monitored_metrics)
                    if not blind_spots:
                        reward -= 0.02

        return reward

    def _detect_contradiction(self, svc: Any, root_cause: str) -> str:
        """Return a contradiction description if deep_inspect evidence contradicts root_cause."""
        cause = root_cause.lower()
        # connection/pool hypothesis but connections are actually low
        if any(w in cause for w in ("connection", "pool", "exhaust")):
            ratio = svc.connection_count / max(svc.max_connections, 1)
            if ratio < 0.4:
                return f"connection_count is only {svc.connection_count}/{svc.max_connections} (low — not exhausted)"
        # disk hypothesis but disk is fine
        if any(w in cause for w in ("disk", "storage", "space")):
            if svc.disk_usage < 50:
                return f"disk_usage is {svc.disk_usage:.1f}% (not a disk issue)"
        # data corruption hypothesis but integrity is fine
        if any(w in cause for w in ("corrupt", "integrity", "data")):
            if svc.data_integrity > 0.95:
                return f"data_integrity is {svc.data_integrity:.3f} (no corruption detected)"
        # memory hypothesis but memory is fine
        if any(w in cause for w in ("memory", "leak", "oom")):
            if svc.memory_usage < 50:
                return f"memory_usage is {svc.memory_usage:.1f}% (memory is healthy)"
        return ""

    def _do_state_hypothesis(self, params: Dict[str, Any]) -> Tuple[float, str]:
        root_cause = params.get("root_cause", "").strip()
        confidence = float(params.get("confidence", 0.5))
        reasoning = params.get("reasoning", "").strip()

        if not root_cause or len(root_cause) < 5:
            return 0.0, "state_hypothesis requires root_cause (min 5 chars)."
        if not reasoning or len(reasoning) < 10:
            return 0.0, "state_hypothesis requires reasoning (min 10 chars)."

        confidence = max(0.0, min(1.0, confidence))

        self._active_hypothesis = {
            "root_cause": root_cause,
            "confidence": confidence,
            "reasoning": reasoning,
            "step": self._steps_taken,
        }
        self._hypothesis_history.append(dict(self._active_hypothesis))
        self._contradiction_pending = False

        return 0.0, (
            f"┌─────────────────────────────────────────────────────┐\n"
            f"│  HYPOTHESIS STATED                                    │\n"
            f"├─────────────────────────────────────────────────────┤\n"
            f"│  Root cause:  {root_cause[:60]}\n"
            f"│  Confidence:  {confidence:.0%}\n"
            f"│  Reasoning:   {reasoning[:80]}\n"
            f"├─────────────────────────────────────────────────────┤\n"
            f"│  Use deep_inspect to gather evidence.\n"
            f"│  If evidence contradicts — use revise_hypothesis().\n"
            f"│  When confident — use commit_root_cause().\n"
            f"└─────────────────────────────────────────────────────┘"
        )

    def _do_revise_hypothesis(self, params: Dict[str, Any]) -> Tuple[float, str]:
        root_cause = params.get("root_cause", "").strip()
        new_confidence = float(params.get("new_confidence", 0.5))
        reason = params.get("reason", "").strip()

        if not root_cause or len(root_cause) < 5:
            return 0.0, "revise_hypothesis requires root_cause (min 5 chars)."
        if not reason or len(reason) < 10:
            return 0.0, "revise_hypothesis requires reason (min 10 chars)."

        new_confidence = max(0.0, min(1.0, new_confidence))

        # Core reward signal: did the agent revise BECAUSE of a contradiction?
        reward = 0.0
        revision_msg = ""
        if self._contradiction_pending:
            reward = 0.03
            self._revisions_after_contradiction += 1
            self._state.hypothesis_revisions += 1
            revision_msg = "\n  ✓ Revised after contradicting evidence — correct epistemic update: +0.03"
            self._contradiction_pending = False
        elif self._active_hypothesis is None:
            reward = -0.01
            revision_msg = "\n  No active hypothesis to revise. Use state_hypothesis first. (-0.01)"
        else:
            # Revision without contradiction — neutral, allowed
            revision_msg = "\n  Hypothesis revised (no pending contradiction detected)."

        self._active_hypothesis = {
            "root_cause": root_cause,
            "confidence": new_confidence,
            "reasoning": reason,
            "step": self._steps_taken,
            "revised": True,
        }
        self._hypothesis_history.append(dict(self._active_hypothesis))

        return reward, (
            f"┌─────────────────────────────────────────────────────┐\n"
            f"│  HYPOTHESIS REVISED                                   │\n"
            f"├─────────────────────────────────────────────────────┤\n"
            f"│  New root cause:  {root_cause[:55]}\n"
            f"│  New confidence:  {new_confidence:.0%}\n"
            f"│  Reason:          {reason[:75]}\n"
            f"└─────────────────────────────────────────────────────┘"
            f"{revision_msg}"
        )

    def _do_commit_root_cause(self, params: Dict[str, Any]) -> Tuple[float, str]:
        root_cause = params.get("root_cause", "").strip()
        evidence_summary = params.get("evidence_summary", "").strip()

        if not root_cause or len(root_cause) < 5:
            return 0.0, "commit_root_cause requires root_cause (min 5 chars)."
        if not evidence_summary or len(evidence_summary) < 10:
            return 0.0, "commit_root_cause requires evidence_summary (min 10 chars)."

        self._state.commits_total += 1

        # Penalize committing with low confidence or without inspecting anything
        reward = 0.0
        commit_msg = ""

        active_conf = self._active_hypothesis.get("confidence", 0.0) if self._active_hypothesis else 0.0
        has_evidence = len(self._inspected_services) > 0

        if not has_evidence:
            reward = -0.03
            self._state.premature_commits += 1
            commit_msg = "\n  ✗ Committed without inspecting any service — premature commit. (-0.03)"
        elif active_conf < 0.5 and self._active_hypothesis is not None:
            reward = -0.02
            self._state.premature_commits += 1
            commit_msg = f"\n  ✗ Committed with low confidence ({active_conf:.0%}) — gather more evidence first. (-0.02)"
        else:
            reward = 0.02
            commit_msg = "\n  ✓ Root cause committed with sufficient evidence. (+0.02)"

        self._committed_root_causes.append({
            "root_cause": root_cause,
            "evidence_summary": evidence_summary,
            "confidence": active_conf,
            "step": self._steps_taken,
        })

        return reward, (
            f"┌─────────────────────────────────────────────────────┐\n"
            f"│  ROOT CAUSE COMMITTED                                 │\n"
            f"├─────────────────────────────────────────────────────┤\n"
            f"│  Root cause:  {root_cause[:60]}\n"
            f"│  Evidence:    {evidence_summary[:75]}\n"
            f"│  Confidence:  {active_conf:.0%}\n"
            f"└─────────────────────────────────────────────────────┘"
            f"{commit_msg}"
        )

    def _do_classify_finding(self, params: Dict[str, Any]) -> Tuple[float, str]:
        finding_type = params.get("finding_type", "").strip().lower()
        severity = params.get("severity", "medium").strip().lower()
        is_silent = bool(params.get("is_silent", False))
        affected = params.get("affected_services", [])
        root_cause = params.get("root_cause", "")
        evidence = params.get("evidence", "")

        if not finding_type:
            return 0.0, (
                "classify_finding requires parameters.finding_type.\n"
                f"Valid types: {', '.join(sorted(VALID_FINDING_TYPES))}"
            )

        # ── Reward Hacking Prevention ──────────────────────────────

        # Anti-spam: same finding type submitted more than twice
        type_count = self._finding_types_submitted.get(finding_type, 0)
        if type_count >= 2:
            return -0.05, (
                f"Finding type '{finding_type}' already submitted {type_count} times.\n"
                "Submitting the same finding type repeatedly is penalized. (-0.05)"
            )
        self._finding_types_submitted[finding_type] = type_count + 1

        # Coherence check: affected services must overlap with services acted on
        if isinstance(affected, str):
            affected = [s.strip() for s in affected.split(",") if s.strip()]
        affected_set = set(s.lower().strip() for s in affected)

        if affected_set and self._services_acted_on:
            acted_lower = set(s.lower() for s in self._services_acted_on)
            # Also include inspected services (agent may have found via deep_inspect)
            inspected_lower = set(s.lower() for s in self._inspected_services)
            reachable = acted_lower | inspected_lower
            overlap = affected_set & reachable
            if not overlap:
                return -0.03, (
                    f"Coherence check failed: none of {list(affected_set)} were\n"
                    f"acted on or inspected during this audit. (-0.03)\n"
                    f"Services you interacted with: {sorted(reachable)}"
                )

        if severity not in VALID_SEVERITIES:
            severity = "medium"

        type_warning = ""
        if finding_type not in VALID_FINDING_TYPES:
            type_warning = (
                f"\n  Note: '{finding_type}' is not a standard finding type. "
                f"Standard types score higher."
            )

        finding = {
            "finding_type": finding_type,
            "severity": severity,
            "is_silent": is_silent,
            "affected_services": affected,
            "root_cause": root_cause,
            "evidence": evidence,
        }

        reward, matched_vuln = self._grade_finding(finding)
        finding["reward_earned"] = reward
        finding["matched_vulnerability"] = matched_vuln
        self._findings.append(finding)
        self._state.total_findings = len(self._findings)

        if is_silent:
            self._state.silent_failures_found += 1
        else:
            self._state.loud_failures_found += 1

        if reward > 0.20:
            quality = "EXCELLENT — strong match to a known vulnerability"
        elif reward > 0.10:
            quality = "GOOD — partial match found"
        elif reward > 0.03:
            quality = "PARTIAL — some elements match"
        else:
            quality = "WEAK — no strong match to known vulnerabilities"

        return reward, (
            f"Finding classified: {finding_type}\n"
            f"  Severity: {severity} | Silent: {is_silent}\n"
            f"  Affected: {', '.join(affected) if affected else 'not specified'}\n"
            f"  Assessment: {quality} (reward: {reward:.3f}){type_warning}"
        )

    def _grade_finding(self, finding: Dict[str, Any]) -> Tuple[float, str]:
        best_score = 0.0
        best_vuln = ""

        for vuln in self._scenario.vulnerabilities:
            if vuln.name in self._matched_vulns:
                continue
            score = self._match_vulnerability(finding, vuln)
            if score > best_score:
                best_score = score
                best_vuln = vuln.name

        if best_score > 0.05 and best_vuln:
            self._matched_vulns.add(best_vuln)

        return round(best_score, 4), best_vuln

    def _match_vulnerability(
        self, finding: Dict[str, Any], vuln: GroundTruthVulnerability
    ) -> float:
        f_type = finding.get("finding_type", "").lower().strip()
        v_type = vuln.finding_type.lower()

        type_score = 0.0
        if f_type == v_type:
            type_score = 1.0
        else:
            f_words = set(w for w in f_type.split("_") if len(w) > 3)
            v_words = set(w for w in v_type.split("_") if len(w) > 3)
            overlap = len(f_words & v_words)
            if overlap >= 2:
                type_score = 0.6
            elif overlap == 1:
                type_score = 0.25

        f_services = set(s.lower().strip() for s in finding.get("affected_services", []))
        v_services = set(s.lower() for s in vuln.affected_services)
        service_score = 0.0
        if f_services and v_services:
            intersection = len(f_services & v_services)
            union = len(f_services | v_services)
            service_score = intersection / max(union, 1)

        if type_score == 0.0 and service_score == 0.0:
            return 0.0

        score = 0.0
        score += vuln.weight * 0.40 * type_score
        score += vuln.weight * 0.20 * service_score

        if finding.get("severity", "").lower() == vuln.severity.lower():
            score += vuln.weight * 0.15

        if finding.get("is_silent", False) == vuln.is_silent:
            score += vuln.weight * 0.15

        root = finding.get("root_cause", "").lower()
        if root:
            key_concepts = set(svc.lower() for svc in vuln.affected_services)
            technical_terms = {
                "data_integrity", "connection", "pool", "corrupt", "replica",
                "propagat", "monitor", "silent", "alert", "threshold",
                "circuit", "breaker", "quorum", "latency", "timeout",
                "queue", "overflow", "partition", "memory", "disk",
                "cascade", "exhaustion", "stale", "masking",
            }
            gt_root = vuln.root_cause.lower()
            relevant_terms = {t for t in technical_terms if t in gt_root}

            matched_concepts = sum(1 for c in key_concepts if c in root)
            matched_terms = sum(1 for t in relevant_terms if t in root)
            total_expected = len(key_concepts) + len(relevant_terms)

            if total_expected > 0:
                concept_score = (matched_concepts + matched_terms) / total_expected
                score += vuln.weight * 0.10 * min(1.0, concept_score)

        return score

    def _do_submit_report(self) -> Tuple[float, str]:
        if not self._findings:
            return 0.001, (
                "┌─────────────────────────────────────┐\n"
                "│       AUDIT REPORT SUBMITTED         │\n"
                "├─────────────────────────────────────┤\n"
                "│  Findings: 0                         │\n"
                "│  No vulnerabilities documented.      │\n"
                "│  FINAL SCORE: 0.001                  │\n"
                "└─────────────────────────────────────┘"
            )

        finding_score = sum(f.get("reward_earned", 0) for f in self._findings)

        total_budget = self._scenario.chaos_budget
        used = self._state.chaos_budget_used
        efficiency_bonus = (
            0.08 if used <= total_budget * 0.5
            else 0.03 if used <= total_budget * 0.75
            else 0.0
        )

        stealth_bonus = 0.08 if self._graph.total_alerts_fired == 0 else 0.0

        # Inference mastery bonus — reward agents that reason before looking
        inference_bonus = 0.0
        if self._infer_attempts >= 2 and self._infer_correct / max(self._infer_attempts, 1) >= 0.6:
            inference_bonus = 0.05

        false_findings = sum(1 for f in self._findings if f.get("reward_earned", 0) <= 0.01)
        false_penalty = false_findings * 0.05

        total = max(
            0.0,
            finding_score + efficiency_bonus + stealth_bonus + inference_bonus - false_penalty,
        )
        max_possible = sum(v.weight for v in self._scenario.vulnerabilities) + 0.21
        score = total / max(max_possible, 0.01)
        score = max(0.001, min(0.999, score))

        self._update_metrics()

        lines = [
            "┌─────────────────────────────────────────────────────┐",
            "│              AUDIT REPORT SUBMITTED                  │",
            "├─────────────────────────────────────────────────────┤",
            f"│  Total findings: {len(self._findings):<35}│",
            f"│  Vulnerabilities matched: {len(self._matched_vulns)}/{len(self._scenario.vulnerabilities):<26}│",
            f"│  Silent failures found: {self._state.silent_failures_found:<28}│",
            "├─────────────────────────────────────────────────────┤",
            f"│  Finding score:         {finding_score:>8.3f}                   │",
            f"│  Efficiency bonus:     +{efficiency_bonus:>8.3f}  ({used}/{total_budget} budget)    │",
            f"│  Stealth bonus:        +{stealth_bonus:>8.3f}  ({self._graph.total_alerts_fired} alerts)      │",
            f"│  Inference bonus:      +{inference_bonus:>8.3f}  ({self._infer_correct}/{self._infer_attempts} correct)  │",
            f"│  False finding penalty: -{false_penalty:>7.3f}  ({false_findings} false)       │",
            "├─────────────────────────────────────────────────────┤",
            f"│  Stealth Ratio:         {self._state.stealth_ratio:>8.3f}                   │",
            f"│  Obs Gap Exploit Rate:  {self._state.obs_gap_exploit_rate:>8.3f}                   │",
            f"│  Inference Accuracy:    {self._state.infer_accuracy:>8.3f}                   │",
            f"│  Hypothesis Revisions:  {self._state.hypothesis_revisions:>8d}  ({self._contradiction_events} contradictions) │",
            f"│  Revision Rate:         {self._state.revision_rate:>8.3f}                   │",
            f"│  Premature Commits:     {self._state.premature_commits:>8d}                   │",
            "├─────────────────────────────────────────────────────┤",
            f"│  FINAL SCORE:           {score:>8.3f}                   │",
            "└─────────────────────────────────────────────────────┘",
        ]
        return score, "\n".join(lines)

    # ── Helpers ─────────────────────────────────────────────────────

    def _make_observation(
        self,
        reward: float,
        action_result: str,
        done: bool = False,
    ) -> SystemObservation:
        return SystemObservation(
            done=done,
            reward=reward,
            services=self._graph.get_monitoring_view() if self._graph else {},
            alerts=self._graph.get_alerts_summary() if self._graph else [],
            action_result=action_result,
            system_description="",
            monitoring_status=(
                self._graph.get_monitoring_status() if self._graph else ""
            ),
            chaos_budget_remaining=self._chaos_budget,
            steps_remaining=self._max_steps - self._steps_taken,
            findings=[
                {
                    "type": f.get("finding_type", ""),
                    "severity": f.get("severity", ""),
                    "silent": f.get("is_silent", False),
                    "reward": round(f.get("reward_earned", 0), 3),
                }
                for f in self._findings
            ],
            task_name=self._state.task_name,
        )
