# Tau Bench training and evaluation

This directory contains the canonical Tau Airline/Retail entry points for
Agentic OPD, outcome-GRPO, and native evaluation. Machine-specific paths and
service endpoints are intentionally not stored here.

## Protocol

- Tau is pinned to commit `17e07b1da2bbc0cadfddeea36412686e0604127b`
  plus the checked-in optional-voice compatibility patch.
- Training uses the complete official `train` split: Airline 30 and Retail 74.
  There is no qualification or expert-success filter.
- Every formal optimizer step contains 16 task groups: 5 Airline and 11 Retail.
- Agentic OPD uses four same-state student candidates and a K=3 teacher
  multiset, then commits one uniform-argmax candidate.
- Outcome-GRPO uses four independent full trajectories per task and terminal
  trajectory reward. It never enters the teacher/state-group path.
- Student and teacher receive the same native Tau conversation and tool schemas.
  Privileged teacher context is opt-in and disabled by default.
- Final evaluation uses Tau's native runner and deterministic
  ENV/ACTION/COMMUNICATE criteria.

Training, in-process validation, and standalone evaluation limits are
independent: 20 agent decisions, 30 agent decisions, and 200 native simulation
steps by default.

## Setup

Tau requires Python 3.12 or newer. By default the checkout is placed beside this
repository at `../tau2-bench`; override `TAU2_ROOT` when needed.

```bash
PYTHON=python TAU2_ROOT=/path/to/tau2-bench \
  bash examples/tau_bench/install_tau2.sh
```

The installer pins Tau, applies the compatibility patch, and installs
`tau2[gym,knowledge]` editable into `PYTHON`.

## Runtime configuration

The launchers contain protocol defaults, not cluster identity. Configure these
values in your shell or in an untracked environment file.

| Variable | Meaning |
|---|---|
| `MODEL_PATH` | Student model directory; required |
| `PYTHON` | Python executable; defaults to `python` |
| `TAU2_ROOT` | Pinned Tau checkout; defaults to sibling `../tau2-bench` |
| `TAU_USER_MODEL` | LiteLLM identifier for the served user model, e.g. `openai/qwen3.5-9b`; required |
| `TAU_USER_API_BASE` | OpenAI-compatible user endpoint; required for training |
| `TAU_USER_API_KEY` | User endpoint key; defaults to `EMPTY` |
| `TAU_TEACHER_MODEL` | Raw model ID served by the teacher endpoint; required by Agentic OPD |
| `TAU_TEACHER_API_BASE` | OpenAI-compatible teacher endpoint; required by Agentic OPD |
| `TAU_TEACHER_API_KEY` | Teacher endpoint key; defaults to `EMPTY` |

For an OpenAI-compatible vLLM user endpoint, retain the `openai/` LiteLLM
prefix in `TAU_USER_MODEL`; the teacher client uses the raw served model ID.

The validated sampling protocol is still Qwen-oriented: the teacher uses
thinking with temperature 0.6, top-p 0.95, top-k 20 and 8,192 output tokens;
the Qwen3.5 user simulator uses thinking with temperature 1.0, top-p 0.95,
top-k 20, presence penalty 1.5 and 8,192 output tokens. Override these only as
an explicit protocol change.

Current cluster values and ready-to-source profiles are documented separately
in `docs/temp_docs/agentic_opd_docs/cluster_runtime.md`.

## Training

After exporting the runtime configuration above:

```bash
# Agentic OPD
METHOD=agentic_opd N_GPUS=2 \
RUN_DIR="runs/$(date -u +%Y%m%dT%H%M%SZ)-tau-agentic-opd" \
bash examples/tau_bench/train/run.sh

# Outcome-GRPO (does not require teacher variables)
METHOD=outcome N_GPUS=2 \
RUN_DIR="runs/$(date -u +%Y%m%dT%H%M%SZ)-tau-outcome" \
bash examples/tau_bench/train/run.sh
```

Set `N_GPUS=8` for the validated eight-GPU profile. Other GPU counts require
explicit TP, SP, PPO-token, and log-prob-token settings.

| GPUs | rollout TP | actor SP | PPO/log-prob tokens per GPU |
|---:|---:|---:|---:|
| 2 | 1 | 2 | 16,384 |
| 8 | 2 | 4 | 8,192 |

Formal defaults are 100 optimizer steps, checkpoint every 10 steps, no
step-zero validation, and no periodic test evaluation. Use a separate native
evaluation run for reported results.

### Smoke

```bash
METHOD=agentic_opd N_GPUS=2 SMOKE=1 RUN_DIR=/tmp/tau-opd-smoke \
  bash examples/tau_bench/train/run.sh

METHOD=outcome N_GPUS=2 SMOKE=1 RUN_DIR=/tmp/tau-outcome-smoke \
  bash examples/tau_bench/train/run.sh
```

### Resume

Resume into the original run directory so task schedule, manifest, cache,
TensorBoard series, and checkpoint numbering remain aligned.

```bash
METHOD=agentic_opd N_GPUS=2 \
RUN_DIR=/absolute/path/to/original-run \
RESUME_MODE=resume_path \
RESUME_FROM_PATH=/absolute/path/to/original-run/ckpt/global_step_10 \
bash examples/tau_bench/train/run.sh
```

The training environment seed is currently fixed to zero. Repeated launches
must not be reported as controlled independent training seeds until a
first-class seed interface is added.

## Native evaluation

Copy `examples/tau_bench/eval/models.example.tsv` and list each base model or
VERL checkpoint. Then run:

```bash
MODEL_SPECS_FILE=/path/to/models.tsv \
RUN_DIR=runs/tau_native_eval \
USER_SIMULATOR_MODE=remote \
TAU_USER_MODEL=openai/served-user-model-name \
TAU_USER_API_BASE=http://user-host:port/v1 \
TAU_USER_API_KEY=... \
bash examples/tau_bench/eval/run.sh
```

Defaults are the complete official `test` split (Airline 20, Retail 40),
three trials, greedy agent decoding, and resumable task-sharded outputs.

If no remote endpoint is configured, `USER_SIMULATOR_MODE=auto` selects local
mode. Local mode reserves physical GPU 0 for the user simulator and serves the
evaluated agent on all remaining GPUs:

```bash
MODEL_SPECS_FILE=/path/to/models.tsv \
USER_MODEL_PATH=/path/to/user-model \
USER_VLLM_BIN=/path/to/vllm \
bash examples/tau_bench/eval/run.sh
```

An explicitly configured but unreachable remote endpoint fails loudly rather
than silently changing the evaluation protocol. The evaluator never stops an
unrelated GPU process. Use `NUM_TASKS=1 NUM_TRIALS=1 DOMAINS=airline` for a
bounded smoke test and `DRY_RUN=1` to inspect a launch without serving models.

## Implementation status

Implemented and smoke-tested here:

- Agentic OPD training;
- outcome-GRPO training;
- official-split preparation;
- native resumable evaluation with remote or local user simulation.

Teacher-trajectory SeqKD, full-vocabulary reverse-KL OPD, and sampled
reverse-KL OPD remain planned baselines; they are not exposed by
`examples/tau_bench/train/run.sh`.

## Layout

- `train/run.sh`: canonical Agentic OPD/outcome launcher.
- `train/prepare_data.py`: deterministic official-split builder.
- `eval/run.sh`: native evaluator and serving lifecycle.
- `eval/native_eval.py`: Tau runner, manifests, retries, and summaries.
- `install_tau2.sh`: pinned source and dependency installer.
