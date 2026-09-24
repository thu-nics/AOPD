"""Atomic FSDP-to-Hugging-Face export of trusted, local checkpoints."""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from aopd.data import sha256
from aopd.launch import ROOT


def checkpoint_identity(checkpoint):
    actor = Path(checkpoint).resolve()
    if (actor / "actor").is_dir():
        actor /= "actor"
    shards = sorted(actor.glob("model_world_size_*_rank_*.pt"))
    sizes = {int(re.fullmatch(r"model_world_size_(\d+)_rank_\d+\.pt", p.name)[1]) for p in shards}
    if len(sizes) != 1:
        raise ValueError(f"incomplete or ambiguous FSDP checkpoint: {actor}")
    size = sizes.pop()
    if {p.name for p in shards} != {f"model_world_size_{size}_rank_{rank}.pt" for rank in range(size)}:
        raise ValueError(f"incomplete FSDP checkpoint: {actor}")
    # Content hashes deliberately ignore timestamps (shared-storage copies may round them).
    files = shards + sorted((actor / "huggingface").glob("*")) + sorted(actor.glob("*.json")) + sorted(actor.glob("*.jinja"))
    return {"files": {str(p.relative_to(actor)): sha256(p) for p in files if p.is_file()}}


def export_checkpoint(checkpoint, target, *, python=sys.executable):
    checkpoint, target = Path(checkpoint).resolve(), Path(target).resolve()
    actor = checkpoint / "actor" if (checkpoint / "actor").is_dir() else checkpoint
    identity = checkpoint_identity(checkpoint)
    if target.exists():
        manifest = target / "export_manifest.json"
        if not manifest.is_file():
            raise ValueError("export target exists without verified identity; refusing overwrite")
        previous = json.loads(manifest.read_text())
        if previous.get("checkpoint") != identity or any(sha256(target / p) != digest for p, digest in previous["files"].items()):
            raise ValueError("export identity mismatch; choose a new output directory")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".export-", dir=target.parent))
    try:
        subprocess.run([python, str(ROOT / "scripts/model_merger.py"), "merge", "--backend", "fsdp", "--local_dir", str(actor), "--target_dir", str(temporary)], check=True)
        if not (temporary / "config.json").is_file() or not list(temporary.glob("*.safetensors")) or not (temporary / "tokenizer.json").is_file():
            raise ValueError("merger produced an incomplete Hugging Face model")
        if checkpoint_identity(checkpoint) != identity:
            raise ValueError("checkpoint identity changed during export")
        manifest = {"checkpoint": identity, "files": {p.name: sha256(p) for p in temporary.iterdir() if p.is_file()}}
        (temporary / "export_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        os.rename(temporary, target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return target
