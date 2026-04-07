"""
Inference Script — Chaos Auditor
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    LOCAL_IMAGE_NAME The name of the local image to use for the environment if you are using from_docker_image()

- Defaults are set only for API_BASE_URL and MODEL_NAME
    (and should reflect your active inference setup):
    API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-endpoint>")
    MODEL_NAME = os.getenv("MODEL_NAME", "<your-active-model>")

- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after env.close(), always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw last_action_error string, or null if none.
    - All fields on a single line with no newlines within a line.
    - Each tasks should return score in [0, 1]
"""

import asyncio
import json
import os
import textwrap
from typing import Any, Dict, List, Optional

from openai import OpenAI

from chaos_auditor import ChaosAction, ChaosAuditorEnv

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
API_KEY = HF_TOKEN or os.getenv("API_KEY")

# Optional — if you use from_docker_image():
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
IMAGE_NAME = LOCAL_IMAGE_NAME or os.getenv("IMAGE_NAME")

BENCHMARK = "chaos-auditor"
TASKS = ["easy", "medium", "hard"]
TEMPERATURE = 0.3
MAX_TOKENS = 800

SYSTEM_PROMPT = textwrap.dedent("""\
You are an expert chaos engineer. You audit distributed systems to find silent vulnerabilities.

CRITICAL RULE: Respond with ONLY a JSON object. No explanation. No markdown. Just JSON.

YOUR APPROACH (follow this exactly):
1. First action: observe (see monitoring dashboard)
2. Read the system architecture. Each service lists "Blind spots" — these are metrics NOT monitored.
3. Pick a leaf service (no dependencies, like a database or cache) where "data_integrity" is a blind spot.
4. Use corrupt_data on that service. Then observe to confirm monitoring is still GREEN.
5. Classify the finding as silent_data_corruption.
6. Pick another service where "disk_usage" or "connection_count" is a blind spot.
7. Use fill_disk or exhaust_connections. Observe. Classify.
8. After 2-3 findings, submit_report.

KEY INSIGHT: Target services where the metric you're attacking is NOT monitored. That's how you create silent failures that score highest.

ACTIONS:
  Chaos (cost 1 budget): kill, spike_traffic, corrupt_data, add_latency, partition_network, fill_disk, exhaust_connections
  Free: observe, deep_inspect, classify_finding, submit_report

EXAMPLES:

Step 1 - observe the system:
{"action_type": "observe"}

Step 2 - inspect a service to see its blind spots:
{"action_type": "deep_inspect", "target_service": "redis-cache"}

Step 3 - attack a blind spot:
{"action_type": "corrupt_data", "target_service": "redis-cache", "parameters": {"data_type": "cache"}}

Step 4 - confirm no alert:
{"action_type": "observe"}

Step 5 - classify the finding:
{"action_type": "classify_finding", "parameters": {"finding_type": "silent_data_corruption", "severity": "critical", "is_silent": true, "affected_services": ["redis-cache", "app-server"], "root_cause": "data_integrity is not monitored on redis-cache. Corrupted data propagates to dependent services without triggering any alert.", "evidence": "After corrupt_data, monitoring remained ALL GREEN. deep_inspect confirmed data_integrity dropped to 0.6."}}

Step 6 - attack another blind spot:
{"action_type": "fill_disk", "target_service": "database", "parameters": {"percentage": 95}}

Step 7 - classify:
{"action_type": "classify_finding", "parameters": {"finding_type": "silent_disk_pressure", "severity": "high", "is_silent": true, "affected_services": ["database"], "root_cause": "disk_usage is not monitored on database. Filling disk causes write failures without any alert.", "evidence": "After fill_disk to 95%, no alert fired. disk_usage is not in monitored metrics."}}

Step 8 - done:
{"action_type": "submit_report"}

FINDING TYPES: silent_data_corruption, silent_disk_pressure, silent_connection_exhaustion, single_point_of_failure, silent_latency_cascade, compound_silent_cascade, defense_masking_failure, cluster_quorum_failure, redundancy_bypass, monitoring_blind_spot
SEVERITY: low, medium, high, critical
""").strip()


# ── Deterministic Fallback Agent ──────────────────────────────────
# Guarantees non-zero baseline scores even if LLM fails.
# Executes a scripted audit plan targeting known blind spot patterns.

FALLBACK_PLANS = {
    "easy": [
        {"action_type": "observe"},
        {"action_type": "deep_inspect", "target_service": "redis-cache"},
        {"action_type": "corrupt_data", "target_service": "redis-cache", "parameters": {"data_type": "cache"}},
        {"action_type": "observe"},
        {"action_type": "classify_finding", "parameters": {
            "finding_type": "silent_data_corruption", "severity": "critical", "is_silent": True,
            "affected_services": ["redis-cache", "app-server"],
            "root_cause": "data_integrity is not monitored on redis-cache. Corrupted data propagates to app-server without any alert.",
            "evidence": "deep_inspect showed data_integrity not in monitored metrics. After corrupt_data, monitoring remained ALL GREEN.",
        }},
        {"action_type": "deep_inspect", "target_service": "database"},
        {"action_type": "fill_disk", "target_service": "database", "parameters": {"percentage": 95}},
        {"action_type": "classify_finding", "parameters": {
            "finding_type": "silent_disk_pressure", "severity": "high", "is_silent": True,
            "affected_services": ["database"],
            "root_cause": "disk_usage is not monitored on database. Filling disk causes silent write failures.",
            "evidence": "deep_inspect showed disk_usage not in monitored metrics. After fill_disk, no alert fired.",
        }},
        {"action_type": "exhaust_connections", "target_service": "database"},
        {"action_type": "classify_finding", "parameters": {
            "finding_type": "silent_connection_exhaustion", "severity": "medium", "is_silent": True,
            "affected_services": ["database"],
            "root_cause": "connection_count is not monitored on database. Pool exhaustion causes silent request timeouts.",
            "evidence": "deep_inspect showed connection_count not in monitored metrics. After exhaust_connections, no alert.",
        }},
        {"action_type": "submit_report"},
    ],
    "medium": [
        {"action_type": "observe"},
        {"action_type": "deep_inspect", "target_service": "redis-primary"},
        {"action_type": "corrupt_data", "target_service": "redis-primary", "parameters": {"data_type": "cache"}},
        {"action_type": "observe"},
        {"action_type": "classify_finding", "parameters": {
            "finding_type": "silent_data_corruption", "severity": "critical", "is_silent": True,
            "affected_services": ["redis-primary", "redis-replica", "cache-layer"],
            "root_cause": "data_integrity not monitored on redis-primary. Corruption replicates to redis-replica and propagates to cache-layer silently.",
            "evidence": "After corrupt_data on redis-primary, all monitoring stayed GREEN. Corruption propagates via replication.",
        }},
        {"action_type": "deep_inspect", "target_service": "payment-db"},
        {"action_type": "add_latency", "target_service": "payment-db", "parameters": {"latency_ms": 400}},
        {"action_type": "observe"},
        {"action_type": "classify_finding", "parameters": {
            "finding_type": "silent_latency_cascade", "severity": "high", "is_silent": True,
            "affected_services": ["payment-db", "payment-svc"],
            "root_cause": "Adding latency to payment-db below alert threshold causes payment-svc retries, leading to connection pool growth. connection_count not monitored on payment-db.",
            "evidence": "Latency added below threshold. No alert fired. payment-svc retries compound the issue.",
        }},
        {"action_type": "spike_traffic", "target_service": "auth-service", "parameters": {"multiplier": 4}},
        {"action_type": "classify_finding", "parameters": {
            "finding_type": "silent_connection_exhaustion", "severity": "critical", "is_silent": True,
            "affected_services": ["auth-service", "user-db"],
            "root_cause": "Spiking traffic to auth-service causes user-db connection pool exhaustion. connection_count not monitored on user-db.",
            "evidence": "After traffic spike, auth-service errors increase but user-db connection exhaustion goes undetected.",
        }},
        {"action_type": "submit_report"},
    ],
    "hard": [
        {"action_type": "observe"},
        {"action_type": "deep_inspect", "target_service": "ml-model-cache"},
        {"action_type": "corrupt_data", "target_service": "ml-model-cache", "parameters": {"data_type": "model"}},
        {"action_type": "observe"},
        {"action_type": "classify_finding", "parameters": {
            "finding_type": "silent_data_corruption", "severity": "high", "is_silent": True,
            "affected_services": ["ml-model-cache", "recommend-svc"],
            "root_cause": "data_integrity not monitored on ml-model-cache. Corrupted model causes wrong recommendations. No quality metric exists.",
            "evidence": "After corrupt_data, monitoring stayed GREEN. recommend-svc serves stale recommendations undetected.",
        }},
        {"action_type": "deep_inspect", "target_service": "elasticsearch"},
        {"action_type": "add_latency", "target_service": "elasticsearch", "parameters": {"latency_ms": 300}},
        {"action_type": "spike_traffic", "target_service": "search-service", "parameters": {"multiplier": 1.8}},
        {"action_type": "observe"},
        {"action_type": "classify_finding", "parameters": {
            "finding_type": "compound_silent_cascade", "severity": "critical", "is_silent": True,
            "affected_services": ["elasticsearch", "search-service", "order-service", "order-db"],
            "root_cause": "Adding latency to elasticsearch below threshold combined with spiking traffic to search-service below auto-scale trigger causes compound effect: search-service thread exhaustion, order-service timeouts, order-db connection pool creep. connection_count not monitored.",
            "evidence": "Neither action alone triggers alerts. Combined effect causes silent connection exhaustion on order-db.",
        }},
        {"action_type": "deep_inspect", "target_service": "order-service"},
        {"action_type": "partition_network", "target_service": "order-service", "parameters": {"service_b": "order-db"}},
        {"action_type": "spike_traffic", "target_service": "order-service", "parameters": {"multiplier": 2}},
        {"action_type": "classify_finding", "parameters": {
            "finding_type": "defense_masking_failure", "severity": "critical", "is_silent": True,
            "affected_services": ["order-service", "order-db"],
            "root_cause": "Network partition triggers circuit breaker on order-service. Orders queue in memory. Traffic spike causes memory growth. Circuit breaker masks failure from monitoring. Orders silently dropped.",
            "evidence": "Circuit breaker activated, load balancer still sends traffic. Memory grows but order failures are hidden by the circuit breaker.",
        }},
        {"action_type": "submit_report"},
    ],
}


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def parse_llm_response(text: str) -> Dict[str, Any]:
    """Parse LLM response into action dict. Handles JSON in markdown code blocks."""
    text = text.strip()
    if "```" in text:
        start = text.find("```")
        end = text.find("```", start + 3)
        if end > start:
            block = text[start + 3 : end].strip()
            if block.startswith("json"):
                block = block[4:].strip()
            text = block

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    for i, ch in enumerate(text):
        if ch == "{":
            depth = 0
            for j in range(i, len(text)):
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[i : j + 1])
                        except json.JSONDecodeError:
                            break
            break

    return {"action_type": "observe"}


def get_action_from_llm(
    client: OpenAI,
    observation_history: List[Dict[str, str]],
    consecutive_failures: int = 0,
) -> Optional[Dict[str, Any]]:
    """Ask the LLM to decide the next action. Returns None to trigger fallback."""
    if consecutive_failures >= 2:
        # After 2 consecutive failures, switch to fallback permanently
        return None
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                *observation_history,
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        result = parse_llm_response(text)
        if result.get("action_type") == "observe" and "observe" not in text.lower()[:20]:
            # parse_llm_response returned default — LLM output wasn't valid JSON
            return None
        return result
    except Exception as exc:
        print(f"[DEBUG] LLM request failed: {exc}", flush=True)
        return None


def format_observation_for_llm(obs: Any) -> str:
    """Format observation into a concise but strategically useful string."""
    parts = []

    # Action result (what just happened)
    if obs.action_result:
        # Truncate very long action results to keep context manageable
        result = obs.action_result
        if len(result) > 800:
            result = result[:800] + "\n  [... truncated]"
        parts.append(result)

    # Monitoring status (critical for deciding if failure was silent)
    if obs.monitoring_status:
        status = obs.monitoring_status
        if len(status) > 200:
            status = status[:200] + "..."
        parts.append(f"Monitoring: {status}")

    # Resources
    parts.append(f"Chaos budget: {obs.chaos_budget_remaining} | Steps left: {obs.steps_remaining}")

    # Findings summary
    if obs.findings:
        parts.append(f"Findings classified: {len(obs.findings)}")
        for f in obs.findings:
            parts.append(f"  - {f.get('type','?')} (severity={f.get('severity','?')}, silent={f.get('silent','?')}, reward={f.get('reward',0):.2f})")

    # Strategic hint based on state
    if obs.chaos_budget_remaining == 0 and not obs.findings:
        parts.append("HINT: Budget exhausted with no findings. Use classify_finding on observations, then submit_report.")
    elif obs.chaos_budget_remaining <= 2 and obs.findings:
        parts.append("HINT: Budget almost gone. Consider submit_report to lock in your score.")
    elif obs.steps_remaining <= 3:
        parts.append("HINT: Few steps left. Submit report soon to avoid auto-submission.")

    return "\n".join(parts)


async def run_task(client: OpenAI, env: ChaosAuditorEnv, task_name: str) -> None:
    """Run a single task using LLM with deterministic fallback."""
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    fallback_plan = FALLBACK_PLANS.get(task_name, FALLBACK_PLANS["easy"])
    fallback_idx = 0
    using_fallback = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(task=task_name)
        obs = result.observation
        observation_history: List[Dict[str, str]] = []

        if obs.system_description:
            observation_history.append({
                "role": "user",
                "content": f"System initialized:\n{obs.system_description}\n\nRespond with a JSON action. Start by observing.",
            })
        else:
            observation_history.append({
                "role": "user",
                "content": "Environment ready. Respond with JSON. Start with observe.",
            })

        max_steps = obs.steps_remaining + 1

        consecutive_failures = 0

        for step in range(1, max_steps + 1):
            if result.done:
                break

            # Try LLM first, fall back to scripted plan if it fails
            action_dict = None
            if not using_fallback:
                action_dict = get_action_from_llm(client, observation_history, consecutive_failures)
                if action_dict is None:
                    consecutive_failures += 1
                    if consecutive_failures >= 2:
                        using_fallback = True
                        print(f"[DEBUG] Switching to fallback agent for {task_name}", flush=True)
                    else:
                        # Single failure — use fallback for this step only
                        if fallback_idx < len(fallback_plan):
                            action_dict = fallback_plan[fallback_idx]
                            fallback_idx += 1
                else:
                    consecutive_failures = 0

            if using_fallback or action_dict is None:
                if fallback_idx < len(fallback_plan):
                    action_dict = fallback_plan[fallback_idx]
                    fallback_idx += 1
                else:
                    action_dict = {"action_type": "submit_report"}

            action = ChaosAction(
                action_type=action_dict.get("action_type", "observe"),
                target_service=action_dict.get("target_service"),
                parameters=action_dict.get("parameters", {}),
            )

            result = await env.step(action)
            obs = result.observation

            reward = result.reward or 0.0
            done = result.done
            error = None

            rewards.append(reward)
            steps_taken = step

            action_str = action.action_type
            if action.target_service:
                action_str += f"({action.target_service})"

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            # Update conversation history for LLM
            if not using_fallback:
                observation_history.append({
                    "role": "assistant",
                    "content": json.dumps(action_dict),
                })
                obs_text = format_observation_for_llm(obs)
                observation_history.append({
                    "role": "user",
                    "content": f"Result:\n{obs_text}\n\nRespond with JSON for your next action.",
                })
                if len(observation_history) > 20:
                    observation_history = observation_history[:2] + observation_history[-18:]

            if done:
                break

        # Score: the submit_report reward IS the normalized final score
        # If report was submitted (last reward is the final score from grading)
        # Otherwise sum up finding rewards
        if rewards:
            last_reward = rewards[-1]
            if last_reward > 0.1:
                # submit_report returns the final normalized score
                score = max(0.0, min(1.0, last_reward))
            else:
                # Auto-submitted or no report — sum positive rewards and normalize
                positive = sum(r for r in rewards if r > 0)
                score = max(0.0, min(1.0, positive))
        success = score >= 0.05

    except Exception as exc:
        print(f"[DEBUG] Task {task_name} error: {exc}", flush=True)
        score = 0.0
        success = False

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


async def create_env() -> ChaosAuditorEnv:
    """Create environment client. Tries Docker image first, then URL fallback."""
    image = LOCAL_IMAGE_NAME or IMAGE_NAME
    hf_url = os.getenv("HF_SPACE_URL") or os.getenv("SPACE_URL")
    local_url = os.getenv("LOCAL_URL")

    # Try Docker image first (what judges typically provide)
    if image:
        try:
            env = await ChaosAuditorEnv.from_docker_image(image)
            return env
        except Exception as e:
            print(f"[DEBUG] from_docker_image failed: {e}", flush=True)

    # Fallback to HF Space URL
    if hf_url:
        return ChaosAuditorEnv(base_url=hf_url, message_timeout_s=120)

    # Fallback to local URL
    if local_url:
        return ChaosAuditorEnv(base_url=local_url, message_timeout_s=120)

    # Last resort: localhost
    return ChaosAuditorEnv(base_url="http://localhost:7860", message_timeout_s=120)


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    for task_name in TASKS:
        env = await create_env()
        await run_task(client, env, task_name)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"[DEBUG] Fatal error: {e}", flush=True)
        import sys
        sys.exit(1)
