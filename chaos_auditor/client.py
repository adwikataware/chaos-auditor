from typing import Any, Dict

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from chaos_auditor.models import AuditState, ChaosAction, SystemObservation


class ChaosAuditorEnv(EnvClient[ChaosAction, SystemObservation, AuditState]):
    """WebSocket client for the Chaos Auditor environment."""

    def _step_payload(self, action: ChaosAction) -> Dict[str, Any]:
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[SystemObservation]:
        obs_data = payload.get("observation", payload)
        observation = SystemObservation(
            done=payload.get("done", obs_data.get("done", False)),
            reward=payload.get("reward", obs_data.get("reward")),
            services=obs_data.get("services", {}),
            alerts=obs_data.get("alerts", []),
            action_result=obs_data.get("action_result", ""),
            system_description=obs_data.get("system_description", ""),
            monitoring_status=obs_data.get("monitoring_status", ""),
            chaos_budget_remaining=obs_data.get("chaos_budget_remaining", 0),
            steps_remaining=obs_data.get("steps_remaining", 0),
            findings=obs_data.get("findings", []),
            task_name=obs_data.get("task_name", ""),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward", obs_data.get("reward")),
            done=payload.get("done", obs_data.get("done", False)),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> AuditState:
        return AuditState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_name=payload.get("task_name", ""),
            total_findings=payload.get("total_findings", 0),
            chaos_budget_used=payload.get("chaos_budget_used", 0),
            chaos_budget_max=payload.get("chaos_budget_max", 0),
            silent_failures_found=payload.get("silent_failures_found", 0),
            loud_failures_found=payload.get("loud_failures_found", 0),
            current_score=payload.get("current_score", 0.0),
        )
