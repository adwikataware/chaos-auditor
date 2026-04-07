"""
Core environment logic for the Chaos Auditor.

Implements the OpenEnv Environment interface: reset(), step(), state.
Wires together the simulation engine, scenarios, and grading.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional, Set

from openenv.core.env_server import Environment

from chaos_auditor.models import AuditState, ChaosAction, SystemObservation
from chaos_auditor.server.scenarios import (
    GroundTruthVulnerability,
    Scenario,
    get_scenario,
)
from chaos_auditor.server.simulation import ServiceGraph


# Enumerated finding types for structured matching
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
FREE_ACTIONS = {"observe", "deep_inspect", "classify_finding", "submit_report"}
ALL_ACTIONS = CHAOS_ACTIONS | FREE_ACTIONS


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
        self._matched_vulns: Set[str] = set()  # Prevent double-dipping
        self._step_rewards: List[float] = []
        self._done = False
        self._chaos_budget = 0
        self._max_steps = 0
        self._steps_taken = 0

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
            f"  Free: observe, deep_inspect, classify_finding, submit_report\n\n"
            f"FINDING TYPES (use these exact values in classify_finding):\n"
            f"  {finding_types_str}\n\n"
            f"SEVERITY LEVELS: low, medium, high, critical\n\n"
            f"TIP: Use 'observe' to see what monitoring shows. Use 'deep_inspect'\n"
            f"  to see ALL metrics including blind spots. Compare the two to\n"
            f"  identify what monitoring is missing.\n"
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

        # Edge case: invalid action type
        if action_type not in ALL_ACTIONS:
            self._step_rewards.append(0.0)
            return self._make_observation(
                reward=0.0,
                action_result=(
                    f"Unknown action '{action.action_type}'.\n"
                    f"Valid actions: {', '.join(sorted(ALL_ACTIONS))}"
                ),
            )

        # Edge case: chaos action with no budget
        if action_type in CHAOS_ACTIONS:
            if self._chaos_budget <= 0:
                self._step_rewards.append(0.0)
                return self._make_observation(
                    reward=0.0,
                    action_result=(
                        "Chaos budget exhausted (0 remaining). "
                        "Use observe, deep_inspect, classify_finding, or submit_report."
                    ),
                )
            self._chaos_budget -= 1
            self._state.chaos_budget_used += 1

        # Execute action
        reward = 0.0
        action_result = ""

        if action_type == "observe":
            action_result = self._do_observe()

        elif action_type == "deep_inspect":
            action_result = self._do_deep_inspect(action.target_service)

        elif action_type == "classify_finding":
            reward, action_result = self._do_classify_finding(action.parameters)

        elif action_type == "submit_report":
            reward, action_result = self._do_submit_report()
            self._done = True

        else:
            # All chaos actions
            action_result = self._do_chaos(action_type, action.target_service, action.parameters)

        # Tick simulation (system evolves between steps)
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

        # Auto-submit at max steps
        if self._steps_taken >= self._max_steps and not self._done:
            self._done = True
            final_reward, final_msg = self._do_submit_report()
            reward += final_reward
            action_result += f"\n\nMAX STEPS REACHED — Auto-submitting report.\n{final_msg}"

        self._step_rewards.append(reward)
        self._state.current_score = sum(self._step_rewards)

        return self._make_observation(
            reward=reward,
            action_result=action_result,
            done=self._done,
        )

    @property
    def state(self) -> AuditState:
        return self._state

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
            f"Note: This shows ONLY monitored metrics. Use deep_inspect(<service>)\n"
            f"to reveal all metrics including blind spots."
        )
        return "\n".join(lines)

    def _do_deep_inspect(self, target: Optional[str]) -> str:
        if not target:
            services = ", ".join(sorted(self._graph.services.keys()))
            return (
                f"deep_inspect requires target_service parameter.\n"
                f"Available services: {services}"
            )
        svc = self._graph.get_service(target)
        if not svc:
            services = ", ".join(sorted(self._graph.services.keys()))
            return f"Service '{target}' not found. Available: {services}"

        view = svc.get_deep_view()
        lines = [
            f"┌─────────────────────────────────────────────────────┐",
            f"│  DEEP INSPECTION: {target:<33} │",
            f"├─────────────────────────────────────────────────────┤",
        ]
        for k, v in view.items():
            if k == "name":
                continue
            # Flag metrics that are NOT monitored (the blind spots)
            is_blind = k.replace("connections", "connection_count").replace(
                "response_time", "response_time_ms"
            ).replace("request_queue", "request_queue_depth")
            blind_flag = ""
            base_metric = k.split(" ")[0]
            if base_metric in ("data_integrity", "connections", "response_time",
                              "request_queue", "disk_usage", "memory_usage"):
                metric_name = {
                    "data_integrity": "data_integrity",
                    "connections": "connection_count",
                    "response_time": "response_time_ms",
                    "request_queue": "request_queue_depth",
                    "disk_usage": "disk_usage",
                    "memory_usage": "memory_usage",
                }.get(base_metric, base_metric)
                if metric_name not in svc.monitored_metrics:
                    blind_flag = " ◀ NOT MONITORED"

            lines.append(f"│  {k}: {v}{blind_flag}")

        lines.append(f"├─────────────────────────────────────────────────────┤")
        lines.append(f"│  Monitored metrics: {', '.join(svc.monitored_metrics)}")
        blind_spots = {"cpu_usage", "memory_usage", "error_rate", "response_time_ms",
                      "connection_count", "disk_usage", "data_integrity",
                      "request_queue_depth"} - set(svc.monitored_metrics)
        lines.append(f"│  BLIND SPOTS: {', '.join(sorted(blind_spots))}")
        lines.append(f"└─────────────────────────────────────────────────────┘")
        return "\n".join(lines)

    def _do_chaos(self, action_type: str, target: Optional[str], params: Dict[str, Any]) -> str:
        if action_type == "partition_network":
            svc_b = params.get("service_b", "")
            if not target or not svc_b:
                return (
                    "partition_network requires:\n"
                    "  target_service: first service name\n"
                    "  parameters.service_b: second service name"
                )
            return self._graph.partition_network(target, svc_b)

        if not target:
            services = ", ".join(sorted(self._graph.services.keys()))
            return (
                f"{action_type} requires target_service parameter.\n"
                f"Available services: {services}"
            )
        if not self._graph.get_service(target):
            services = ", ".join(sorted(self._graph.services.keys()))
            return f"Service '{target}' not found. Available: {services}"

        dispatch = {
            "kill": lambda: self._graph.kill(target),
            "spike_traffic": lambda: self._graph.spike_traffic(target, float(params.get("multiplier", 3.0))),
            "corrupt_data": lambda: self._graph.corrupt_data(target, params.get("data_type", "cache")),
            "add_latency": lambda: self._graph.add_latency(target, int(params.get("latency_ms", 200))),
            "fill_disk": lambda: self._graph.fill_disk(target, float(params.get("percentage", 95))),
            "exhaust_connections": lambda: self._graph.exhaust_connections(target),
        }
        handler = dispatch.get(action_type)
        if handler:
            return handler()
        return f"Unknown chaos action: {action_type}"

    def _do_classify_finding(self, params: Dict[str, Any]) -> tuple[float, str]:
        finding_type = params.get("finding_type", "").strip().lower()
        severity = params.get("severity", "medium").strip().lower()
        is_silent = bool(params.get("is_silent", False))
        affected = params.get("affected_services", [])
        root_cause = params.get("root_cause", "")
        evidence = params.get("evidence", "")

        # Validate finding type
        if not finding_type:
            return 0.0, (
                "classify_finding requires parameters.finding_type.\n"
                f"Valid types: {', '.join(sorted(VALID_FINDING_TYPES))}"
            )

        # Warn about non-standard finding type (but still accept)
        type_warning = ""
        if finding_type not in VALID_FINDING_TYPES:
            type_warning = (
                f"\n  Note: '{finding_type}' is not a standard finding type. "
                f"Standard types score higher."
            )

        if severity not in VALID_SEVERITIES:
            severity = "medium"

        if isinstance(affected, str):
            affected = [s.strip() for s in affected.split(",") if s.strip()]

        finding = {
            "finding_type": finding_type,
            "severity": severity,
            "is_silent": is_silent,
            "affected_services": affected,
            "root_cause": root_cause,
            "evidence": evidence,
        }

        # Grade against ground truth (with duplicate protection)
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

    def _grade_finding(self, finding: Dict[str, Any]) -> tuple[float, str]:
        """Grade a finding against ground truth. Returns (score, matched_vuln_name)."""
        best_score = 0.0
        best_vuln = ""

        for vuln in self._scenario.vulnerabilities:
            # Skip already-matched vulnerabilities (no double-dipping)
            if vuln.name in self._matched_vulns:
                continue
            score = self._match_vulnerability(finding, vuln)
            if score > best_score:
                best_score = score
                best_vuln = vuln.name

        # Mark this vulnerability as claimed if score is meaningful
        if best_score > 0.05 and best_vuln:
            self._matched_vulns.add(best_vuln)

        return round(best_score, 4), best_vuln

    def _match_vulnerability(
        self, finding: Dict[str, Any], vuln: GroundTruthVulnerability
    ) -> float:
        """Structured matching between finding and ground truth.

        Requires at least a partial match on finding_type OR affected_services
        to score anything. This prevents completely fake findings from getting
        credit just by guessing severity/silent correctly.
        """
        f_type = finding.get("finding_type", "").lower().strip()
        v_type = vuln.finding_type.lower()

        # 1. Finding type (40% of weight)
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

        # 4. Affected services (20%) — set intersection (Jaccard)
        f_services = set(s.lower().strip() for s in finding.get("affected_services", []))
        v_services = set(s.lower() for s in vuln.affected_services)
        service_score = 0.0
        if f_services and v_services:
            intersection = len(f_services & v_services)
            union = len(f_services | v_services)
            service_score = intersection / max(union, 1)

        # GATE: Must match on type OR services to get any credit
        if type_score == 0.0 and service_score == 0.0:
            return 0.0

        score = 0.0
        score += vuln.weight * 0.40 * type_score
        score += vuln.weight * 0.20 * service_score

        # 2. Severity (15%) — exact match only
        if finding.get("severity", "").lower() == vuln.severity.lower():
            score += vuln.weight * 0.15

        # 3. Silent flag (15%) — exact match
        if finding.get("is_silent", False) == vuln.is_silent:
            score += vuln.weight * 0.15

        # 5. Root cause (10%) — key concept matching
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

    def _do_submit_report(self) -> tuple[float, str]:
        """Calculate final score and end the episode."""
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
        efficiency_bonus = 0.08 if used <= total_budget * 0.5 else 0.03 if used <= total_budget * 0.75 else 0.0

        stealth_bonus = 0.08 if self._graph.total_alerts_fired == 0 else 0.0

        false_findings = sum(1 for f in self._findings if f.get("reward_earned", 0) <= 0.01)
        false_penalty = false_findings * 0.05

        total = max(0.0, finding_score + efficiency_bonus + stealth_bonus - false_penalty)
        max_possible = sum(v.weight for v in self._scenario.vulnerabilities) + 0.16
        # Clamp to strictly (0, 1) — judges require not exactly 0.0 or 1.0
        score = total / max(max_possible, 0.01)
        score = max(0.001, min(0.999, score))

        # Build detailed report
        lines = [
            "┌─────────────────────────────────────────────────────┐",
            "│              AUDIT REPORT SUBMITTED                  │",
            "├─────────────────────────────────────────────────────┤",
            f"│  Total findings: {len(self._findings):<35}│",
            f"│  Vulnerabilities matched: {len(self._matched_vulns)}/{len(self._scenario.vulnerabilities):<26}│",
            f"│  Silent failures found: {self._state.silent_failures_found:<28}│",
            "├─────────────────────────────────────────────────────┤",
            f"│  Finding score:       {finding_score:>8.3f}                     │",
            f"│  Efficiency bonus:   +{efficiency_bonus:>8.3f}  (used {used}/{total_budget} budget)    │",
            f"│  Stealth bonus:      +{stealth_bonus:>8.3f}  ({self._graph.total_alerts_fired} alerts fired)    │",
            f"│  False finding penalty: -{false_penalty:>6.3f}  ({false_findings} false)       │",
            "├─────────────────────────────────────────────────────┤",
            f"│  FINAL SCORE:         {score:>8.3f}                     │",
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
