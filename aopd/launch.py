"""Compile scientific recipes and machine-specific runtime settings, without I/O."""

import os
import sys
from pathlib import Path

from aopd.runtime import active_runtime, load_yaml, resolve_roles, validate_generation, validate_runtime

ROOT = Path(__file__).resolve().parents[1]
RECIPES = ("main", "tau-full", "tau-a1", "tau-a4", "tau-a5", "tau-s1", "tau-s2")


def launch_environment(plan, roles):
    """Do not let an old shell experiment silently override a public recipe."""
    allowed = {
        "PATH",
        "HOME",
        "USER",
        "LOGNAME",
        "SHELL",
        "TMPDIR",
        "LANG",
        "LC_ALL",
        "TZ",
        "LD_LIBRARY_PATH",
        "HF_HOME",
        "HF_HUB_CACHE",
        "TRANSFORMERS_CACHE",
        "XDG_CACHE_HOME",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "SSL_CERT_FILE",
        "REQUESTS_CA_BUNDLE",
    }
    allowed.update(role["api_key_env"] for role in roles.values())
    inherited = {k: v for k, v in os.environ.items() if k in allowed or k.startswith(("NCCL_", "TORCH_NCCL_"))}
    return {**inherited, **plan["env"]}


def scalar(value):
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def role_environment(prefix, role):
    result = {f"{prefix}_{key}": role[field] for key, field in {"PROVIDER": "provider", "MODEL": "model", "API_BASE": "base_url", "API_KEY_ENV": "api_key_env"}.items()}
    for key, value in role["generation"].items():
        result[f"{prefix}_{key.upper()}"] = value
    return result


def build_plan(recipe, runtime, run_dir, *, smoke=False, resume=None):
    """Return argv/env; never allocate GPUs, write files, or probe APIs."""
    if recipe not in RECIPES:
        raise ValueError(f"unknown recipe {recipe}; choose from {RECIPES}")
    if smoke and resume:
        raise ValueError("smoke is a fresh bounded run; use a small explicit training recipe to test resume")
    tau = recipe.startswith("tau-")
    config = load_yaml(ROOT / "configs/recipes" / ("tau-full.yaml" if tau else "main.yaml"))
    env = dict(config["environment"])
    updates = runtime.get("training", {})
    reserved = {"METHOD", "TAU_ABLATION", "TAU_TEACHER_SOURCE", "TAU_USE_PRIVILEGED_TEACHER_CONTEXT", "USE_RAW_SPLIT", "MANAGE_AWM_SERVER", "RESUME_MODE"}
    if set(updates) & reserved:
        raise ValueError(f"reserved recipe controls: {sorted(set(updates) & reserved)}")
    if extra := set(updates) - set(env):
        raise ValueError(f"unknown training settings: {sorted(extra)}")
    env.update(updates)
    student = runtime["student"]
    env.update(
        {
            "MODEL_PATH": runtime["model"],
            "PYTHON": runtime.get("python", sys.executable),
            "RUN_DIR": str(Path(run_dir).resolve()),
            "CUDA_VISIBLE_DEVICES": ",".join(map(str, student["gpus"])),
            "N_GPUS": len(student["gpus"]),
            "TP_SIZE": student.get("tp", 1),
            "SP_SIZE": student.get("sp", 1),
            "PPO_MAX_TOKENS_PER_GPU": student.get("ppo_tokens_per_gpu", 16384),
            "LOGPROB_MAX_TOKENS_PER_GPU": student.get("logprob_tokens_per_gpu", 16384),
            "GPU_MEM_UTIL": student.get("memory_utilization", 0.65),
            "MAX_NUM_BATCHED_TOKENS": student.get("max_batched_tokens", 32768),
        }
    )
    roles = resolve_roles(runtime)
    required = {"user", "matcher"}
    if recipe not in {"tau-s1", "tau-s2"}:
        required.add("teacher")
    if not tau:
        required.update({"runtime_judge", "terminal_judge"})
    if missing := required - roles.keys():
        raise ValueError(f"missing API roles: {sorted(missing)}")
    runtime = active_runtime(runtime, required)
    validate_runtime(runtime)
    roles = resolve_roles(runtime)
    validate_generation(roles, tau=tau)
    sources, data = runtime.get("sources", {}), runtime.get("data", {})
    overrides = []
    for key in ("param_offload", "optimizer_offload"):
        if key in student:
            if not isinstance(student[key], bool):
                raise ValueError(f"student.{key} must be a boolean")
            overrides.append(f"actor_rollout_ref.actor.fsdp_config.{key}={scalar(student[key])}")
    if tau:
        env["TAU2_ROOT"] = sources.get("tau", str(ROOT.parent / "tau2-bench"))
        env["TAU_ABLATION"] = recipe.removeprefix("tau-") if recipe in {"tau-a1", "tau-a4", "tau-a5"} else "full"
        if recipe in {"tau-s1", "tau-s2"}:
            env.update(TAU_TEACHER_SOURCE="self", TAU_USE_PRIVILEGED_TEACHER_CONTEXT=recipe == "tau-s2")
            env["TAU_SELF_CUSTOMER_BRIEFS"] = data.get("customer_briefs", "")
            if recipe == "tau-s2" and not env["TAU_SELF_CUSTOMER_BRIEFS"]:
                raise ValueError("tau-s2 requires data.customer_briefs")
        for name in required:
            env.update(role_environment("TAU_" + name.upper(), roles[name]))
        # Tau's user simulator is routed by LiteLLM, unlike its raw teacher client.
        if not env["TAU_USER_MODEL"].startswith(("openai/", "deepseek/")):
            env["TAU_USER_MODEL"] = "openai/" + env["TAU_USER_MODEL"]
        if "TAU_USER_ENABLE_THINKING" in env:
            env["TAU_USER_REASONING_ENABLED"] = env.pop("TAU_USER_ENABLE_THINKING")
        env["TAU_MATCHER_PROFILE"] = runtime["roles"]["matcher"].get("profile", "default")
    else:
        for role, prefix in {"teacher": "ORACLE", "matcher": "MATCHER", "runtime_judge": "RUNTIME_JUDGE", "terminal_judge": "TERMINAL_JUDGE"}.items():
            env.update(role_environment(prefix, roles[role]))
        for source, key in {"envscaler": "ENVSCALER_ROOT", "awm": "AWM_SOURCE_DIR", "awm_data": "AWM_DATA_DIR"}.items():
            if source in sources:
                env[key] = sources[source]
        for field, key in {"awm_pool": "TRAIN_DATA", "awm_manifest": "TRAIN_SELECTION_MANIFEST", "envscaler_pool": "ENVSCALER_POOL", "envscaler_manifest": "ENVSCALER_MANIFEST"}.items():
            if not data.get(field):
                raise ValueError(f"main requires data.{field}")
            env[key] = data[field]
        user = roles["user"]
        for field, key in {"provider": "provider", "model": "model", "base_url": "api_base", "api_key_env": "api_key_env"}.items():
            overrides.append(f"env.envscaler.user_simulator.{key}={scalar(user[field])}")
        for key, value in user["generation"].items():
            key = "reasoning_enabled" if key == "enable_thinking" else key
            overrides.append(f"env.envscaler.user_simulator.{key}={scalar(value)}")
    if smoke:
        env.update(TRAIN_STEPS=2 if recipe in {"tau-s1", "tau-s2"} else 1, SAVE_FREQ=1, TEST_FREQ=-1)
        if tau:
            env.update(SMOKE=1, SMOKE_TRAIN_STEPS=env["TRAIN_STEPS"], SMOKE_SAVE_FREQ=1, SMOKE_AIRLINE_TRAJ=len(student["gpus"]), SMOKE_RETAIL_TRAJ=0, SMOKE_TELECOM_TRAJ=0, SMOKE_PPO_MINI_BATCH=max(8, 4 * len(student["gpus"])))
        else:
            batch = max(4, len(student["gpus"]))
            env.update(TRAIN_BATCH=batch, AWM_PER_STEP=batch - 1, ENVSCALER_PER_STEP=1, PPO_MINI_BATCH=batch)
            overrides.extend(["env.agentic_mix.bounded_smoke=true", "env.awm.train_max_steps=2", "env.envscaler.train_max_steps=2", "env.max_steps=2"])
    if resume:
        env.update(RESUME_MODE="resume_path", RESUME_FROM_PATH=str(Path(resume).resolve()))
    count, sp = len(student["gpus"]), int(env["SP_SIZE"])
    batch = count if tau and smoke else (sum(int(env[k]) for k in ("AIRLINE_TRAJ", "RETAIL_TRAJ", "TELECOM_TRAJ")) if tau else int(env["TRAIN_BATCH"]))
    if batch <= 0 or batch % count:
        raise ValueError("training task batch must be positive and divisible by student GPU count")
    if not tau and int(env["AWM_PER_STEP"]) + int(env["ENVSCALER_PER_STEP"]) != batch:
        raise ValueError("AWM_PER_STEP + ENVSCALER_PER_STEP must equal TRAIN_BATCH")
    mini = int(env.get("SMOKE_PPO_MINI_BATCH", env["PPO_MINI_BATCH"]))
    if mini <= 0 or mini % (count // sp):
        raise ValueError("PPO_MINI_BATCH must be positive and divisible by student GPU count / SP")
    sequence = 9216 if tau and smoke else sum(int(env[k]) for k in (("MAX_PROMPT", "MAX_RESPONSE") if tau else ("MAX_PROMPT_LENGTH", "MAX_RESPONSE_LENGTH")))
    model_length = (sequence + (4096 if recipe in {"tau-s1", "tau-s2"} else 0)) if tau and smoke else int(env["MAX_MODEL_LEN"])
    if int(env["MAX_NUM_BATCHED_TOKENS"]) < model_length:
        raise ValueError("max_batched_tokens must cover max_model_len for this rollout implementation")
    for key in ("PPO_MAX_TOKENS_PER_GPU", "LOGPROB_MAX_TOKENS_PER_GPU"):
        if int(env[key]) * sp < sequence:
            raise ValueError(f"{key}: token budget * SP must fit {sequence} tokens")
    env = {key: scalar(value) for key, value in env.items()}
    return {"recipe": recipe, "active_roles": sorted(required), "env": env, "command": ["bash", str(ROOT / config["entrypoint"]), *overrides]}
