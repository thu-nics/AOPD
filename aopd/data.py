"""Package/verify fixed pools without modifying their source artifacts."""

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

FILES = {
    "awm": {
        "02_deterministic_audit": ["deterministic_manifest.json", "scenario_health.jsonl", "task_audit.jsonl"],
        "03_static_feasibility_judge": ["health_manifest.json", "config.json", "task_audit.jsonl", "awm_training_pool.parquet"],
    },
    "envscaler": {
        "01_deterministic_audit": ["deterministic_manifest.json", "task_audit.jsonl"],
        "02_static_feasibility_judge": ["health_manifest.json", "config.json", "task_audit.jsonl", "envscaler_training_pool.parquet"],
    },
}

# The frozen command identifies one release, independently of bundle.json.
# Pin the complete evidence chain as well as the pools and reviewed train-only
# briefs. Custom datasets remain supported through explicit runtime data paths.
RELEASE_FILE_SHA256 = {
    "awm/02_deterministic_audit/deterministic_manifest.json": "fa5e85ced390537088b94c414703eaf30937b79a91de2bc4a7c615a64ff55841",
    "awm/02_deterministic_audit/scenario_health.jsonl": "5de769cde347df5a39561d945c9b414a51b9f0d2ef36f7817abb0d433e47d08e",
    "awm/02_deterministic_audit/task_audit.jsonl": "8ef106431f11ac2cc7bbfb8c0864b1af5f5eb7e0eddd1519fdc47709c9e2b025",
    "awm/03_static_feasibility_judge/awm_training_pool.parquet": "86e7580a7b41459a805c41b6212babea10a4ce41462e9586466f6d173beceb46",
    "awm/03_static_feasibility_judge/config.json": "85e1335ed1381178a7259ae44986fb88a28849d4aec16d0386a6fcbc7bb91769",
    "awm/03_static_feasibility_judge/health_manifest.json": "a4be3047c629b5c23af15350fd11125f3b5a68154b6007bf4125ebc2bb6c77c6",
    "awm/03_static_feasibility_judge/task_audit.jsonl": "7cd371e2d64148d35bcb3d20cc1bfa73e06bacdf7b418ff6a142988de3ecec7b",
    "envscaler/01_deterministic_audit/deterministic_manifest.json": "3b267c26247762745fe7c656dfb2f9f15d1b3391515a138cc9f8f66608599765",
    "envscaler/01_deterministic_audit/task_audit.jsonl": "ddab41a5de802329d85fcb9c39661f5f61c5873d41d8e1c25d31e3629fe66936",
    "envscaler/02_static_feasibility_judge/config.json": "134ec5ec581faa31f95a86f0eaefe09b4dccaa9f65eb395a5f36b26201e962bc",
    "envscaler/02_static_feasibility_judge/envscaler_training_pool.parquet": "464c6d1f890aa447e2ec2a20d5b6b53dc0f608fd2914cf091fb87968f7b49a4f",
    "envscaler/02_static_feasibility_judge/health_manifest.json": "45b0ae3d8be5219e9b4742f4a25042aa4f6f4f8c78d63b11a2017b8860fe33b8",
    "envscaler/02_static_feasibility_judge/task_audit.jsonl": "3558303ebeb6e21ac20f4ecc709771e9d54c7dd1c4199b0ac9984b72b0c1b8ff",
    "tau/customer_briefs.json": "a3523035cb7d53097389f52e83f1a1db21e09c78a7a7dce15d817659f98673e5",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def bundle(research_runs, output, briefs):
    if output.exists():
        raise ValueError("bundle output must not exist; source artifacts are never overwritten")
    inputs = [(research_runs / f"{domain}_data_processing" / stage / name, Path(domain) / stage / name) for domain, stages in FILES.items() for stage, names in stages.items() for name in names]
    inputs.append((briefs, Path("tau/customer_briefs.json")))
    for source, _ in inputs:
        if not source.is_file():
            raise FileNotFoundError(source)
    original_hashes = {}
    for source, relative in inputs:
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        original_hashes[str(relative)] = sha256(source)
    # Absolute source locations are provenance, not dataset identity.
    deterministic = output / "envscaler/01_deterministic_audit/deterministic_manifest.json"
    config = output / "envscaler/02_static_feasibility_judge/config.json"
    health = config.parent / "health_manifest.json"
    for path in (deterministic, config, health):
        value = json.loads(path.read_text())
        value["source"]["source_root"] = "EnvScaler"
        if path == config:
            value["deterministic_manifest_sha256"] = sha256(deterministic)
        if path == health:
            value["artifacts"]["config"]["sha256"] = sha256(config)
        _json(path, value)
    _json(output / "bundle.json", {"protocol": "aopd-fixed-pools-v1", "original_sha256": original_hashes, "files": {str(relative): sha256(output / relative) for _, relative in inputs}})
    return verify(output)


def verify_release_files(output, manifest):
    """Require exact released membership and bytes, even if hashes are rewritten."""
    if manifest.get("protocol") != "aopd-fixed-pools-v1":
        raise ValueError("unknown fixed data bundle protocol")
    files = manifest.get("files")
    if not isinstance(files, dict) or files.keys() != RELEASE_FILE_SHA256.keys():
        raise ValueError("fixed bundle file set must include every released pool, audit and training brief")
    for relative, expected in RELEASE_FILE_SHA256.items():
        if files[relative] != expected:
            raise ValueError(f"fixed bundle differs from released identity: {relative}")
        target = (output / relative).resolve()
        if not target.is_relative_to(output.resolve()) or not target.is_file() or sha256(target) != expected:
            raise ValueError(f"bundle integrity failure: {relative}")


def verify(output):
    manifest = json.loads((output / "bundle.json").read_text())
    verify_release_files(output, manifest)
    from agent_system.environments.env_package.awm.data.pools import verify_training_pool

    awm = output / "awm/03_static_feasibility_judge"
    verify_training_pool(awm / "awm_training_pool.parquet", awm / "health_manifest.json")
    from agent_system.environments.env_package.envscaler.data import materialize_mixed_schedule
    from agent_system.environments.env_package.tau_bench.customer_briefs import BriefStore

    envscaler = output / "envscaler/02_static_feasibility_judge"
    with tempfile.TemporaryDirectory(prefix="aopd-data-check-") as temporary:
        materialize_mixed_schedule(
            awm_data=awm / "awm_training_pool.parquet",
            envscaler_data=envscaler / "envscaler_training_pool.parquet",
            envscaler_manifest=envscaler / "health_manifest.json",
            output_data=Path(temporary) / "schedule.parquet",
            output_manifest=Path(temporary) / "schedule.json",
            train_steps=1,
            awm_per_step=1,
            envscaler_per_step=1,
        )
    briefs = BriefStore(output / "tau/customer_briefs.json")
    return {"files_verified": len(manifest["files"]), "reviewed_customer_briefs": len(briefs.records)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    pack = sub.add_parser("bundle")
    pack.add_argument("--research-runs", type=Path, required=True)
    pack.add_argument("--briefs", type=Path, required=True)
    pack.add_argument("--output", type=Path, required=True)
    check = sub.add_parser("verify")
    check.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(bundle(args.research_runs, args.output, args.briefs) if args.command == "bundle" else verify(args.output)))


if __name__ == "__main__":
    main()
