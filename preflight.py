"""
Pre-flight check — run before burning HF compute credits.
Validates: env starts, all 6 hypothesis actions work, contradiction fires,
deep_inspect triggers blind-spot discovery, submit_report returns non-zero score.
Takes ~10 seconds, no GPU needed.
"""
import sys, os
sys.stdout = __import__("io").TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

PASS = "[PASS]"
FAIL = "[FAIL]"

def check(label, condition, detail=""):
    if condition:
        print(f"  {PASS}  {label}")
    else:
        print(f"  {FAIL}  {label}" + (f" — {detail}" if detail else ""))
    return condition

all_ok = True

print("\n=== Chaos Auditor Pre-flight ===\n")

# 1. Import
try:
    from chaos_auditor.server.environment import ChaosAuditorEnvironment
    from chaos_auditor.models import ChaosAction
    all_ok &= check("imports ok", True)
except Exception as e:
    check("imports ok", False, str(e))
    sys.exit(1)

# 2. Reset all tasks
for task in ["easy", "medium", "hard", "random"]:
    try:
        env = ChaosAuditorEnvironment()
        obs = env.reset(task=task, seed=0)
        all_ok &= check(f"reset({task})", obs.task_name == task, f"got {obs.task_name}")
    except Exception as e:
        all_ok &= check(f"reset({task})", False, str(e))

# 3. Full workflow on easy
env = ChaosAuditorEnvironment()
obs = env.reset(task="easy", seed=42)
svcs = list(env._graph.services.keys())
cache = next((s for s in svcs if any(k in s for k in ["cache","redis"])), svcs[0])
db    = next((s for s in svcs if any(k in s for k in ["db","database"])), svcs[-1])

def step(action_type, target=None, **params):
    return env.step(ChaosAction(action_type=action_type, target_service=target, parameters=params))

obs = step("observe")
all_ok &= check("observe", "MONITORING" in obs.action_result or "green" in obs.action_result.lower())

obs = step("state_hypothesis",
           root_cause="connection pool exhaustion on database",
           confidence=0.6,
           reasoning="response_time rising without cpu spike")
all_ok &= check("state_hypothesis", obs.reward is not None and obs.reward >= 0.0,
                f"reward={obs.reward}")

obs = step("infer_state", cache,
           metric="data_integrity", predicted_state="low",
           reasoning="cache rarely monitors data_integrity metric")
all_ok &= check("infer_state (pre-inspect)", obs.reward == 0.0 or obs.reward == -0.01,
                f"reward={obs.reward}")

obs = step("deep_inspect", cache)
all_ok &= check("deep_inspect returns blind spots", "BLIND SPOTS" in obs.action_result,
                obs.action_result[:120])

# Force a contradiction scenario: set connection_count low then state hypothesis about connections
env2 = ChaosAuditorEnvironment()
env2.reset(task="easy", seed=1)
svcs2 = list(env2._graph.services.keys())
db2 = next((s for s in svcs2 if any(k in s for k in ["db","database"])), svcs2[-1])
env2._graph.services[db2].connection_count = 5  # force low
env2.step(ChaosAction(action_type="state_hypothesis", parameters={
    "root_cause": "connection pool exhaustion on database service",
    "confidence": 0.7,
    "reasoning": "elevated response time suggests connection pool is nearly full"
}))
obs2 = env2.step(ChaosAction(action_type="deep_inspect", target_service=db2))
contradiction_fired = "CONTRADICTION" in obs2.action_result
all_ok &= check("contradiction detection fires", contradiction_fired,
                "CONTRADICTION not found in deep_inspect result")

# revise after contradiction
obs2 = env2.step(ChaosAction(action_type="revise_hypothesis", parameters={
    "root_cause": "disk pressure causing silent write failures",
    "new_confidence": 0.8,
    "reason": "connection_count low — contradicts pool exhaustion hypothesis"
}))
all_ok &= check("revise_hypothesis earns +0.03", abs((obs2.reward or 0) - 0.03) < 0.001,
                f"reward={obs2.reward}")

# commit with evidence
obs = step("commit_root_cause",
           root_cause="data integrity not monitored on cache",
           evidence_summary="deep_inspect showed data_integrity as blind spot on cache")
all_ok &= check("commit_root_cause (with evidence)", (obs.reward or 0) >= 0.0,
                f"reward={obs.reward}")

# chaos on blind spot
obs = step("corrupt_data", cache, data_type="cache")
all_ok &= check("corrupt_data executes", obs.reward is not None)

obs = step("classify_finding",
           finding_type="silent_data_corruption", severity="critical",
           is_silent=True, affected_services=[cache],
           root_cause="data_integrity unmonitored", evidence="deep_inspect confirmed")
all_ok &= check("classify_finding non-negative", (obs.reward or 0) >= 0.0,
                f"reward={obs.reward}")

obs = step("submit_report")
all_ok &= check("submit_report returns score > 0", (obs.reward or 0) > 0.0,
                f"reward={obs.reward}")

# 4. Anti-hacking checks
env3 = ChaosAuditorEnvironment()
env3.reset(task="easy", seed=5)
svcs3 = list(env3._graph.services.keys())
obs3 = env3.step(ChaosAction(action_type="classify_finding", parameters={
    "finding_type": "silent_data_corruption", "severity": "critical",
    "is_silent": True, "affected_services": ["nonexistent-svc-xyz"],
    "root_cause": "test", "evidence": "test"
}))
all_ok &= check("coherence gate penalizes uninspected service",
                (obs3.reward or 0) <= 0.0, f"reward={obs3.reward}")

env4 = ChaosAuditorEnvironment()
env4.reset(task="easy", seed=6)
svcs4 = list(env4._graph.services.keys())
svc4 = svcs4[0]
env4.step(ChaosAction(action_type="deep_inspect", target_service=svc4))
for _ in range(3):
    env4.step(ChaosAction(action_type="classify_finding", parameters={
        "finding_type": "silent_data_corruption", "severity": "critical",
        "is_silent": True, "affected_services": [svc4],
        "root_cause": "test spam", "evidence": "spam"
    }))
obs4 = env4.step(ChaosAction(action_type="classify_finding", parameters={
    "finding_type": "silent_data_corruption", "severity": "critical",
    "is_silent": True, "affected_services": [svc4],
    "root_cause": "test spam", "evidence": "spam"
}))
all_ok &= check("anti-spam penalizes repeated finding type",
                (obs4.reward or 0) < 0.0, f"reward={obs4.reward}")

# 5. Random task generates unique services each episode
env_r1 = ChaosAuditorEnvironment(); env_r1.reset(task="random", seed=10)
env_r2 = ChaosAuditorEnvironment(); env_r2.reset(task="random", seed=99)
svcs_r1 = set(env_r1._graph.services.keys())
svcs_r2 = set(env_r2._graph.services.keys())
all_ok &= check("random task produces different episodes", svcs_r1 != svcs_r2,
                f"both had same services: {svcs_r1}")

print()
if all_ok:
    print("=== ALL CHECKS PASSED — safe to train ===\n")
else:
    print("=== SOME CHECKS FAILED — fix before training ===\n")
    sys.exit(1)
