from typing import Any, Dict, List, Optional

from pydantic import Field

from openenv.core.env_server import Action, Observation, State


class ChaosAction(Action):
    """Action the agent takes in the chaos auditor environment.

    Actions fall into three categories:
    - Chaos actions: Deliberately break/degrade services (costs chaos budget)
    - Observation actions: Inspect system state (free)
    - Analysis actions: Record findings and submit report (free)
    """

    action_type: str = Field(
        description=(
            "Type of action: kill, spike_traffic, corrupt_data, add_latency, "
            "partition_network, fill_disk, exhaust_connections, "
            "observe, deep_inspect, infer_state, classify_finding, submit_report"
        )
    )
    target_service: Optional[str] = Field(
        default=None,
        description="Target service name for chaos or inspection actions",
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Action parameters. Examples: "
            "spike_traffic: {multiplier: 5}, "
            "add_latency: {latency_ms: 200}, "
            "corrupt_data: {data_type: 'cache'}, "
            "fill_disk: {percentage: 90}, "
            "partition_network: {service_b: 'other-service'}, "
            "infer_state: {metric: 'connection_count', predicted_state: 'high', "
            "reasoning: 'response_time rising without CPU spike suggests connection exhaustion'}, "
            "classify_finding: {finding_type: str, severity: str, "
            "is_silent: bool, affected_services: list, root_cause: str, evidence: str}"
        ),
    )


class ServiceStatus(Action):
    """Status of a single service as seen by monitoring."""

    name: str = ""
    status: str = "HEALTHY"
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    error_rate: float = 0.0
    response_time_ms: int = 0


class Alert(Action):
    """A monitoring alert that fired."""

    service_name: str = ""
    metric: str = ""
    value: float = 0.0
    threshold: float = 0.0
    message: str = ""


class Finding(Action):
    """A vulnerability finding classified by the agent."""

    finding_type: str = ""
    severity: str = ""
    is_silent: bool = False
    affected_services: List[str] = Field(default_factory=list)
    root_cause: str = ""
    evidence: str = ""
    reward_earned: float = 0.0


class SystemObservation(Observation):
    """What the agent sees after each action."""

    services: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Service statuses as monitoring shows them",
    )
    alerts: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Active monitoring alerts",
    )
    action_result: str = Field(
        default="",
        description="Description of what happened from the last action",
    )
    system_description: str = Field(
        default="",
        description="Architecture description (full on reset, delta on steps)",
    )
    monitoring_status: str = Field(
        default="ALL GREEN",
        description="Overall monitoring status summary",
    )
    chaos_budget_remaining: int = Field(
        default=0,
        description="Number of chaos actions remaining",
    )
    steps_remaining: int = Field(
        default=0,
        description="Steps remaining in this episode",
    )
    findings: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Vulnerabilities the agent has classified so far",
    )
    task_name: str = Field(
        default="",
        description="Current task difficulty level",
    )


class AuditState(State):
    """Internal episode state for the chaos auditor."""

    task_name: str = ""
    total_findings: int = 0
    chaos_budget_used: int = 0
    chaos_budget_max: int = 0
    silent_failures_found: int = 0
    loud_failures_found: int = 0
    current_score: float = 0.0

    # Partial observability reasoning metrics
    stealth_ratio: float = 0.0          # silent damage actions / total chaos actions
    obs_gap_exploit_rate: float = 0.0   # blind-spot targeted actions / total chaos actions
    infer_attempts: int = 0             # total infer_state calls
    infer_correct: int = 0              # correct inferences before deep_inspect
    infer_accuracy: float = 0.0         # infer_correct / infer_attempts
