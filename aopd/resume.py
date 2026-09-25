"""Compare semantic run identity before allocating hardware or changing artifacts."""

import json
import re
from pathlib import Path

from aopd.data import sha256
from aopd.export import checkpoint_identity


def validate_training_checkpoint(checkpoint, world_size):
    actor = Path(checkpoint) / "actor"
    shards = list(actor.glob("model_world_size_*_rank_*.pt"))
    saved_sizes = {int(re.fullmatch(r"model_world_size_(\d+)_rank_\d+\.pt", p.name)[1]) for p in shards}
    if saved_sizes != {world_size}:
        raise ValueError("resume requires the same FSDP world size as the checkpoint")
    required = [actor / f"{kind}_world_size_{world_size}_rank_{rank}.pt" for kind in ("model", "optim", "extra_state") for rank in range(world_size)]
    required.append(Path(checkpoint) / "data.pt")
    if any(not p.is_file() or p.stat().st_size == 0 for p in required):
        raise ValueError("incomplete checkpoint: model, optimizer, RNG/scheduler and dataloader states are required")


def model_identity(path):
    root = Path(path)
    if not root.is_dir():
        raise ValueError(f"model must be a local, immutable Hugging Face directory: {root}")
    files = sorted({p for pattern in ("*.json", "*.safetensors", "*.bin", "*.jinja") for p in root.glob(pattern) if p.is_file()})
    if not any(p.suffix in {".bin", ".safetensors"} for p in files):
        raise ValueError(f"no model weights found: {root}")
    return {p.name: sha256(p) for p in files}


def tau_identity(path):
    """Bind executable source and all benchmark data, not just the Git HEAD."""
    root = Path(path)
    files = []
    for relative in ("src/tau2", "data"):
        directory = root / relative
        if not directory.is_dir():
            raise ValueError(f"Tau source identity requires {directory}")
        selected = [p for p in directory.rglob("*") if p.is_file() and "__pycache__" not in p.relative_to(root).parts and p.suffix not in {".pyc", ".pyo"}]
        if not selected:
            raise ValueError(f"Tau source identity cannot use an empty {directory}")
        files.extend(selected)
    return {str(p.relative_to(root)): sha256(p) for p in sorted(files)}


def run_identity(plan, runtime):
    from aopd.runtime import resolve_roles

    roles = {name: {key: role.get(key) for key in ("provider", "model", "generation")} for name, role in resolve_roles(runtime).items()}
    for name in roles:
        roles[name]["profile"] = runtime["roles"][name].get("profile", "default")
        service = resolve_roles(runtime)[name]
        if service["mode"] == "local":
            roles[name]["weights"] = model_identity(service["model_path"])
            roles[name]["serving"] = {k: service.get(k) for k in ("max_model_len", "tool_parser", "reasoning_parser", "extra_args")}
    result = {"protocol": "aopd-release-v1", "recipe": plan["recipe"], "roles": roles}
    result["model"] = model_identity(runtime["model"])
    if "TAU2_ROOT" in plan["env"]:
        result["tau"] = tau_identity(plan["env"]["TAU2_ROOT"])
    # Paths, placement, total duration and logging frequency are operational.
    operational = {
        "MODEL_PATH",
        "PYTHON",
        "RUN_DIR",
        "CUDA_VISIBLE_DEVICES",
        "N_GPUS",
        "TP_SIZE",
        "SP_SIZE",
        "PPO_MAX_TOKENS_PER_GPU",
        "LOGPROB_MAX_TOKENS_PER_GPU",
        "GPU_MEM_UTIL",
        "MAX_NUM_BATCHED_TOKENS",
        "TRAIN_STEPS",
        "SAVE_FREQ",
        "TEST_FREQ",
        "RESUME_MODE",
        "RESUME_FROM_PATH",
        "TAU2_ROOT",
        "AWM_SOURCE_DIR",
        "AWM_DATA_DIR",
        "ENVSCALER_ROOT",
    }
    result["training"] = {k: v for k, v in plan["env"].items() if k not in operational and not k.endswith(("_API_BASE", "_API_KEY_ENV"))}
    # User role settings are identified above; other Hydra overrides alter
    # scientific behavior, notably the bounded main smoke step limits.
    result["overrides"] = [arg for arg in plan.get("command", [])[2:] if not arg.startswith("env.envscaler.user_simulator.")]
    result["data"] = {k: sha256(v) for k, v in runtime.get("data", {}).items() if v and Path(v).is_file()}
    return result


def check_resume(plan, runtime, run_dir):
    if plan["recipe"] == "main":
        requested_steps = int(plan["env"]["TRAIN_STEPS"])
        launch = run_dir / "launch.json"
        schedule = run_dir / "data/training_schedule_manifest.json"
        recorded_steps = []
        if launch.exists():
            recorded_steps.append(json.loads(launch.read_text())["env"]["TRAIN_STEPS"])
        if schedule.exists():
            recorded_steps.append(json.loads(schedule.read_text())["train_steps"])
        if any(int(steps) != requested_steps for steps in recorded_steps):
            raise ValueError("main schedule TRAIN_STEPS changed; use a new run directory to extend training")
    identity = run_identity(plan, runtime)
    saved = run_dir / "protocol.json"
    if saved.exists() and json.loads(saved.read_text()) != identity:
        raise ValueError("resume protocol identity changed; choose a new run directory")
    if (run_dir / "launch.json").exists() and not saved.exists():
        raise ValueError("resume lacks verified protocol identity; historical runs cannot be resumed here")
    checkpoint = plan["env"].get("RESUME_FROM_PATH")
    if checkpoint:
        checkpoint = Path(checkpoint)
        validate_training_checkpoint(checkpoint, int(plan["env"]["N_GPUS"]))
        checkpoint_identity(checkpoint)
        parent = checkpoint.parent.parent / "protocol.json"
        if not parent.exists() or json.loads(parent.read_text()) != identity:
            raise ValueError("checkpoint resume protocol identity differs or is missing")
    return identity
