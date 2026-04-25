"""
Eval harness — reproducible evaluation with held-out seeds.
Run after training to get the numbers that go in the README table.

Usage:
    python eval_harness.py --mode scripted          # scripted fallback baseline
    python eval_harness.py --mode random_agent      # random action baseline
    python eval_harness.py --mode llm --model-path ./checkpoints/sft_warmup_final
    python eval_harness.py --mode llm --model-path ./chaos-auditor-trained
"""
import argparse, json, sys, os
import numpy as np
sys.stdout = __import__("io").TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# Held-out evaluation seeds — never used during training
EVAL_SEEDS = {
    "easy":   [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
    "medium": [200, 201, 202, 203, 204, 205, 206, 207, 208, 209],
    "hard":   [300, 301, 302, 303, 304, 305, 306, 307, 308, 309],
}

SCRIPTED_PLANS = {
    "easy": [
        {"action_type": "observe"},
        {"action_type": "state_hypothesis", "parameters": {
            "root_cause": "data integrity not monitored on cache",
            "confidence": 0.6, "reasoning": "cache services often skip data_integrity monitoring"
        }},
        {"action_type": "infer_state", "target_key": "cache",
         "parameters": {"metric": "data_integrity", "predicted_state": "low",
                        "reasoning": "cache typically omits data_integrity from monitored metrics"}},
        {"action_type": "deep_inspect", "target_key": "cache"},
        {"action_type": "commit_root_cause", "parameters": {
            "root_cause": "data integrity not monitored on cache",
            "evidence_summary": "deep_inspect confirmed data_integrity is blind spot on cache"
        }},
        {"action_type": "corrupt_data", "target_key": "cache", "parameters": {"data_type": "cache"}},
        {"action_type": "observe"},
        {"action_type": "classify_finding", "parameters": {
            "finding_type": "silent_data_corruption", "severity": "critical", "is_silent": True,
            "affected_services_key": "cache",
            "root_cause": "data_integrity unmonitored — silent corruption propagates",
            "evidence": "deep_inspect confirmed blind spot. No alert after corrupt_data."
        }},
        {"action_type": "infer_state", "target_key": "db",
         "parameters": {"metric": "disk_usage", "predicted_state": "high",
                        "reasoning": "databases accumulate logs; disk_usage often unmonitored"}},
        {"action_type": "deep_inspect", "target_key": "db"},
        {"action_type": "fill_disk", "target_key": "db", "parameters": {"percentage": 95}},
        {"action_type": "classify_finding", "parameters": {
            "finding_type": "silent_disk_pressure", "severity": "high", "is_silent": True,
            "affected_services_key": "db",
            "root_cause": "disk_usage not monitored on db — fill_disk causes silent write failures",
            "evidence": "deep_inspect confirmed blind spot. No alert after fill_disk."
        }},
        {"action_type": "submit_report"},
    ],
}
SCRIPTED_PLANS["medium"] = SCRIPTED_PLANS["easy"]
SCRIPTED_PLANS["hard"]   = SCRIPTED_PLANS["easy"]


def _resolve(key, svcs):
    keywords = {
        "cache": ["cache","redis","memcached"],
        "db":    ["db","database","postgres","mysql","mongo"],
        "api":   ["api","gateway","frontend","web"],
    }
    for kw in keywords.get(key, [key]):
        for s in svcs:
            if kw in s.lower():
                return s
    return svcs[0]


def run_scripted_episode(task: str, seed: int) -> dict:
    from chaos_auditor.server.environment import ChaosAuditorEnvironment
    from chaos_auditor.models import ChaosAction

    env = ChaosAuditorEnvironment()
    obs = env.reset(task=task, seed=seed)
    svcs = list(env._graph.services.keys())
    plan = SCRIPTED_PLANS[task]
    total = 0.0

    for spec in plan:
        target_key = spec.get("target_key")
        target = _resolve(target_key, svcs) if target_key else None
        params = dict(spec.get("parameters", {}))
        if "affected_services_key" in params:
            k = params.pop("affected_services_key")
            params["affected_services"] = [_resolve(k, svcs)]
        obs = env.step(ChaosAction(
            action_type=spec["action_type"],
            target_service=target,
            parameters=params,
        ))
        total += obs.reward or 0.0
        if obs.done:
            break

    state = env.state
    return {
        "reward": total,
        "final_score": obs.reward or 0.0,
        "stealth_ratio": state.stealth_ratio,
        "infer_accuracy": state.infer_accuracy,
        "revision_rate": state.revision_rate,
        "obs_gap_exploit_rate": state.obs_gap_exploit_rate,
    }


def run_random_episode(task: str, seed: int) -> dict:
    import random
    from chaos_auditor.server.environment import ChaosAuditorEnvironment
    from chaos_auditor.models import ChaosAction

    rng = random.Random(seed)
    env = ChaosAuditorEnvironment()
    obs = env.reset(task=task, seed=seed)
    svcs = list(env._graph.services.keys())
    actions = ["observe","deep_inspect","kill","corrupt_data","fill_disk",
               "classify_finding","submit_report"]
    total = 0.0

    for _ in range(15):
        action_type = rng.choice(actions)
        target = rng.choice(svcs) if action_type not in ("observe","classify_finding","submit_report") else None
        params = {}
        if action_type == "classify_finding":
            params = {"finding_type": "monitoring_blind_spot", "severity": "medium",
                      "is_silent": False, "affected_services": [rng.choice(svcs)],
                      "root_cause": "random", "evidence": "random"}
        obs = env.step(ChaosAction(action_type=action_type, target_service=target, parameters=params))
        total += obs.reward or 0.0
        if obs.done:
            break

    state = env.state
    return {
        "reward": total,
        "final_score": obs.reward or 0.0,
        "stealth_ratio": state.stealth_ratio,
        "infer_accuracy": state.infer_accuracy,
        "revision_rate": state.revision_rate,
        "obs_gap_exploit_rate": state.obs_gap_exploit_rate,
    }


def run_llm_episode(task: str, seed: int, model, tokenizer, system_prompt: str) -> dict:
    import torch, re, json as _json
    from chaos_auditor.server.environment import ChaosAuditorEnvironment
    from chaos_auditor.models import ChaosAction

    def parse_action(text):
        try:
            m = re.search(r'\{[^{}]+\}', text, re.DOTALL)
            if m:
                return _json.loads(m.group())
        except Exception:
            pass
        return {"action_type": "observe"}

    env = ChaosAuditorEnvironment()
    obs = env.reset(task=task, seed=seed)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": obs.system_description + "\n\nBegin your audit."},
    ]
    total = 0.0

    for step in range(20):
        if obs.done:
            break
        if step > 0:
            messages.append({"role": "user", "content": (
                f"Result: {obs.action_result}\n"
                f"Budget: {obs.chaos_budget_remaining} | Steps: {obs.steps_remaining}\n"
                f"Alerts: {obs.monitoring_status}"
            )})
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=200, temperature=0.7,
                                 do_sample=True, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        messages.append({"role": "assistant", "content": response})
        act = parse_action(response)
        obs = env.step(ChaosAction(
            action_type=act.get("action_type","observe"),
            target_service=act.get("target_service"),
            parameters=act.get("parameters",{}),
        ))
        total += obs.reward or 0.0

    state = env.state
    return {
        "reward": total,
        "final_score": obs.reward or 0.0,
        "stealth_ratio": state.stealth_ratio,
        "infer_accuracy": state.infer_accuracy,
        "revision_rate": state.revision_rate,
        "obs_gap_exploit_rate": state.obs_gap_exploit_rate,
    }


def summarize(results: list[dict], label: str, task: str):
    keys = ["final_score","stealth_ratio","infer_accuracy","revision_rate","obs_gap_exploit_rate"]
    print(f"\n  {label} — {task} ({len(results)} episodes, held-out seeds)")
    print(f"  {'Metric':<28} {'Mean':>8}  {'Std':>8}  {'Min':>8}  {'Max':>8}")
    print(f"  {'-'*28} {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
    for k in keys:
        vals = [r[k] for r in results]
        print(f"  {k:<28} {np.mean(vals):>8.3f}  {np.std(vals):>8.3f}  {np.min(vals):>8.3f}  {np.max(vals):>8.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["scripted","random_agent","llm"], default="scripted")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--tasks", nargs="+", default=["easy","medium","hard"])
    args = parser.parse_args()

    model, tokenizer, system_prompt = None, None, None
    if args.mode == "llm":
        if not args.model_path:
            print("--model-path required for llm mode"); sys.exit(1)
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            args.model_path, max_seq_length=2048, load_in_4bit=True
        )
        FastLanguageModel.for_inference(model)
        # Load system prompt from notebook config if available
        system_prompt = open(os.path.join(os.path.dirname(__file__),
                             "training","system_prompt.txt")).read() if os.path.exists(
            os.path.join(os.path.dirname(__file__),"training","system_prompt.txt")) else \
            "You are a chaos auditor. Output JSON actions only."

    print(f"\n{'='*60}")
    print(f"  Eval mode: {args.mode}" + (f" | {args.model_path}" if args.model_path else ""))
    print(f"{'='*60}")

    for task in args.tasks:
        seeds = EVAL_SEEDS[task]
        results = []
        for seed in seeds:
            if args.mode == "scripted":
                r = run_scripted_episode(task, seed)
            elif args.mode == "random_agent":
                r = run_random_episode(task, seed)
            else:
                r = run_llm_episode(task, seed, model, tokenizer, system_prompt)
            results.append(r)
            print(f"  {task} seed={seed}: score={r['final_score']:.3f} "
                  f"stealth={r['stealth_ratio']:.3f} "
                  f"infer={r['infer_accuracy']:.3f} "
                  f"revision={r['revision_rate']:.3f}")

        summarize(results, args.mode, task)

        # Save results to JSON for later comparison
        out_path = f"eval_{args.mode}_{task}.json"
        with open(out_path, "w") as f:
            json.dump({"mode": args.mode, "task": task, "seeds": seeds,
                       "results": results}, f, indent=2)
        print(f"\n  Saved to {out_path}")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
