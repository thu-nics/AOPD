# Installation

Use Linux, Python 3.12 and an NVIDIA CUDA-compatible driver. Adjust microbatch
token limits and GPU placement to fit your hardware.
The training and auxiliary-serving environments are intentionally separate.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
export PIP_CONSTRAINT="$PWD/constraints/train.txt"
python -m pip install -e '.[vllm,test]' -c constraints/train.txt
python -m pip install flash-attn==2.8.3.post1 --no-build-isolation
export TRAIN_PYTHON="$PWD/.venv/bin/python"
```

Install only the sources needed for your experiment, into fresh directories:

```bash
export TAU_SOURCE="$PWD/deps/tau2-bench"
PYTHON="$TRAIN_PYTHON" TAU2_ROOT="$TAU_SOURCE" bash examples/tau_bench/install_tau2.sh

# Main AWM/EnvScaler training only:
export AWM_SOURCE="$PWD/deps/openenv-awm"
export AWM_DATA="$PWD/data/awm-source"
PYTHON="$TRAIN_PYTHON" OPENENV_ROOT="$AWM_SOURCE" bash examples/awm/setup/install_awm.sh
export ENVSCALER_SOURCE="$PWD/deps/EnvScaler"
PYTHON="$TRAIN_PYTHON" ENVSCALER_ROOT="$ENVSCALER_SOURCE" bash examples/envscaler/setup/install_envscaler.sh
```

The source installers pin revisions and reject incompatible checkouts. Tau
also checks its compatibility patch. Do not update source revisions between
training and evaluation. See `docs/data.md` to download the AWM source dataset.
Keep `PIP_CONSTRAINT` set for the source installers so their dependencies cannot
silently upgrade the validated training stack. Run `python -m pip check` after
installing them.

For local Qwen3.5/3.8 auxiliary models, create a separate serving environment
with a vLLM build supporting that architecture and set `service.python` to its
interpreter. Unset `PIP_CONSTRAINT` in that separate setup shell (or use
`env -u PIP_CONSTRAINT ...`); the training pins do not apply to it.
The older training vLLM must not be upgraded in place to serve a
new auxiliary architecture. Remote services need no local auxiliary install.

This repository must be installed from a checkout (`pip install -e .`): launch
recipes and shell entrypoints are repository resources, not a standalone wheel
CLI distribution. Keep any existing research installation in its own venv.
