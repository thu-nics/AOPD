# Tau Bench training and evaluation

This directory contains the canonical Tau Airline/Retail training and native
evaluation entry points for Agentic OPD and outcome-GRPO.

## Protocol

- Tau source: `/mnt/public2/yuanhuining/repos/tau2-bench`, pinned to commit
  `17e07b1da2bbc0cadfddeea36412686e0604127b` plus the checked-in optional-voice
  compatibility patch.
- Training data: the complete official `train` split, Airline 30 and Retail 74.
  There is no qualification or expert-success filter.
- Each formal optimizer step contains 16 task groups: 5 Airline and 11 Retail.
  The deterministic schedule cycles through every domain's official train set.
- Agentic OPD samples four student actions per state, retains the K=3 teacher
  multiset, uses frequency-weighted semantic reward with scale 0.5, commits one
  uniform argmax action, and masks equal-reward state groups.
- Outcome-GRPO performs four independent full rollouts per task and normalizes
  terminal trajectory rewards. It never enters the teacher/state-group path.
- Training, in-process validation, and standalone native-eval decision limits are
  independent (20, 30, and 200 by default).
- Student and teacher see the same native Tau chat and actual tool schemas.
  Privileged teacher context is available only through an explicit opt-in and
  defaults to disabled.
- Terminal success uses Tau's deterministic ENV/ACTION/COMMUNICATE evaluation;
  LLM-judged NL assertions are excluded.

The data manifest records the pinned source identity, all official task IDs and
content hashes, split names, domain quotas, and every validation tail row that
cannot form a complete fixed-domain batch.

## Setup

Tau requires Python 3.12 or newer:

```bash
PYTHON=/opt/venvs/verl-agent/bin/python \
bash examples/tau_bench/install_tau2.sh
```

The installer keeps source and dataset cache under the shared Tau checkout and
installs it editable into the selected environment.

## Remote models

Training is remote-only for the teacher and user simulator. Defaults are:

- semantic teacher: `qwen3-32b` at `http://172.27.20.249:8000/v1`;
- user simulator: `openai/qwen3.5-9b` at
  `http://172.27.20.58:8000/v1`.

Both cluster servers default to the placeholder key `local-qwen-server`.
Override `TAU_TEACHER_API_BASE`, `TAU_TEACHER_MODEL`,
`TAU_TEACHER_API_KEY`, `TAU_USER_API_BASE`, `TAU_USER_MODEL`, or
`TAU_USER_API_KEY` for another OpenAI-compatible service. Endpoint health is
checked before Ray starts.

Teacher sampling follows Qwen3's thinking defaults: temperature 0.6, top-p
0.95, top-k 20, min-p 0, thinking enabled, 8,192 output tokens, and parallel
tool calls disabled. The semantic matcher uses the same endpoint with
temperature 0 and thinking disabled. The cache protocol includes endpoint,
model, sampling, prompt, and context-mode identity, so incompatible records are
ignored.

The Qwen3.5 user simulator uses thinking enabled, temperature 1.0, top-p 0.95,
top-k 20, min-p 0, presence penalty 1.5, repetition penalty 1.0, and 8,192
output tokens. Truncated or empty generations are retried twice without
retaining the rejected turn.

## Training

One launcher selects both supported methods:

```bash
# Agentic OPD (default)
N_GPUS=8 \
bash examples/tau_bench/train/run.sh

# Outcome-GRPO
METHOD=outcome N_GPUS=8 \
bash examples/tau_bench/train/run.sh
```

The default student is `/mnt/public2/yuanhuining/models/Qwen3-4B`, training is
100 optimizer steps, checkpoints are saved every 10 steps, and in-process
validation is disabled (`VAL_BEFORE_TRAIN=false`, `TEST_FREQ=-1`). Final
evaluation is run separately with Tau's native runner.

The launcher has validated hardware profiles:

| GPUs | rollout TP | actor SP | PPO/log-prob tokens per GPU |
|---:|---:|---:|---:|
| 2 | 1 | 2 | 16,384 |
| 8 | 2 | 4 | 8,192 |

For another GPU count, explicitly set `TP_SIZE`, `SP_SIZE`,
`PPO_MAX_TOKENS_PER_GPU`, and `LOGPROB_MAX_TOKENS_PER_GPU`. Set `SMOKE=1` for
the bounded one-step development run. Artifacts live under `runs/` unless
`RUN_DIR` is supplied.

## Native evaluation

Create a tab-separated model registry using
`examples/tau_bench/eval/models.example.tsv`, then run:

```bash
MODEL_SPECS_FILE=/path/to/models.tsv \
RUN_DIR=runs/tau_native_eval_final \
bash examples/tau_bench/eval/run.sh
```

Defaults are the complete official `test` split (Airline 20 and Retail 40),
three trials, and greedy agent decoding. Use `TASK_SPLIT=base` only for an
explicit historical comparison.

`USER_SIMULATOR_MODE=auto` is remote-first. If an explicitly configured
`TAU_USER_API_BASE` is unavailable, evaluation fails rather than changing the
protocol. If no endpoint was explicitly supplied and the built-in cluster
endpoint is unavailable, the launcher extracts
`/mnt/public2/yuanhuining/venvs/vllm-nightly-cu129.tar.gz` into `/opt/venvs`
when needed, deploys Qwen3.5-9B on physical GPU 0, and serves the evaluated
agent on every remaining GPU. Force either path with
`USER_SIMULATOR_MODE=remote` or `USER_SIMULATOR_MODE=local`; local mode requires
at least two GPUs.

The runner uses Tau's pinned native `LLMAgent`, structured function calling,
task-sharded result files, infrastructure retries, and resumable manifests.
Useful smoke settings are `NUM_TASKS=1 NUM_TRIALS=1 DOMAINS=airline`.

## Layout

- `train/run.sh`: canonical Agentic OPD/outcome launcher.
- `train/prepare_data.py`: deterministic official-split dataset builder.
- `eval/run.sh`: native multi-model evaluator and serving lifecycle.
- `eval/native_eval.py`: native Tau driver, summaries, and manifests.
- `eval/deterministic_evaluator.py`: deterministic evaluation repair layer.
- `install_tau2.sh`: pinned source/dependency installation.

The reusable environment, action, teacher, cache, user simulator, and manager
logic lives under `agent_system/environments/env_package/tau_bench/`.
