# Runtime configuration

Science and placement are separate. A recipe fixes the experiment; a runtime
YAML names models, endpoints, paths and devices. `${NAME}` requires that
environment variable; missing variables fail immediately. No private cluster
paths, IPs or secret values are supplied by the repository.

Relative paths in runtime YAML resolve from the repository root. Explicit CLI
paths (`--runtime`, `--run-dir`, `--resume`, export paths) resolve from the current
working directory. URLs and served model IDs are not filesystem paths. Source,
model, data and interpreter paths are checked before services start.

For the remote Tau template, set these before running the README commands:

```bash
export STUDENT_MODEL=/path/to/Qwen3-4B
export TRAIN_PYTHON=/path/to/training-venv/bin/python
export TAU_SOURCE=/path/to/tau2-bench
export TEACHER_MODEL=qwen3-32b
export TEACHER_BASE_URL=http://teacher-host:8000/v1
export TEACHER_API_KEY=EMPTY  # Only for an unauthenticated service you control.
export AUX_MODEL=qwen3.8-27b
export AUX_BASE_URL=http://auxiliary-host:8000/v1
export AUX_API_KEY=EMPTY
cp configs/runtime.tau.example.yaml configs/runtime.local.yaml
```

`student.gpus` lists physical GPU IDs (including noncontiguous IDs); `tp` and
`sp` must divide their count. For six training GPUs use, for example,
`gpus: [2,3,4,5,6,7], tp: 2, sp: 2`. Tune `ppo_tokens_per_gpu`,
`logprob_tokens_per_gpu` and `memory_utilization` for the hardware. These
settings do not implicitly change the 24-task Tau or 64-task main collection
batch. `student.optimizer_offload: true` (and optionally `param_offload: true`)
can reduce persistent training memory at the cost of CPU transfers. This is
particularly useful when only one GPU is assigned to a small student: reducing
rollout KV memory alone does not release optimizer state between updates. The
recipes do not enable offload automatically. These settings do not change the
training objective. Resume must retain explicitly selected offload settings.

Six Tau GPUs with SP=2 additionally require `training: {PPO_MINI_BATCH:
24}`. The main 64-task recipe cannot use six training GPUs unchanged: task batch
must divide GPU count, and PPO mini-batch must divide GPU count / SP. `--check`
validates this before starting services. A `training` mapping may explicitly
override keys listed in the selected recipe's `environment`; unknown keys fail.
Main training requires `SHUFFLE: false` to preserve its deterministic mixed
schedule. Configure validation using the section below, not `training.TEST_FREQ`
or `training.VAL_BEFORE_TRAIN`.

Services use `mode: api` or `mode: local`. Local services specify `model_path`,
`python`, `gpus`, `tp`, optional `dp` (default 1), and a unique `port`.
`tp * dp` must equal that service's GPU count. Roles reference a named service;
user and matcher may share one service, but independent services cannot overlap
GPUs. API services have no GPU assignment and are never stopped by this launcher.

```yaml
services:
  auxiliary:
    mode: local
    provider: vllm
    model: qwen3.8-27b
    model_path: ${AUX_MODEL_PATH}
    python: ${SERVE_PYTHON}
    gpus: [0, 1]
    tp: 2
    port: 8101
    max_model_len: 65536
roles:
  user: {service: auxiliary, generation: {enable_thinking: false, temperature: 0.7, top_p: 0.8, max_tokens: 8192}}
  matcher: {service: auxiliary, profile: qwen38_concise, generation: {enable_thinking: false, max_tokens: 32768}}
```

Providers: `vllm`, `deepseek`, `dashscope`, `zai`, `openai-compatible`.
The generic provider sends standard Chat Completions fields, not vendor-specific
thinking controls. Use a named provider for explicit thinking support. Model
listing is required only for local vLLM readiness; generic remote services are
tested by a bounded Chat Completions request.

To change the main teacher to Qwen3.7-Flash, change only the teacher service to
`provider: dashscope` and the corresponding model/base URL/key environment name;
set teacher generation to `enable_thinking: true, temperature: 0.6, top_p: 0.95,
max_tokens: 8192`. Keep auxiliary roles on their explicitly selected service.

Run one launcher per experiment with disjoint GPU lists and unique service
ports. To share a service across experiments, launch it independently and give
both experiments `mode: api`; neither experiment owns its lifetime.

Both AOPD recipes require user, teacher and matcher roles. Main training also
requires runtime-judge and terminal-judge roles. Unsupported generation keys
fail explicitly rather than being silently ignored.

Export an FSDP checkpoint for an external benchmark runner:

```bash
python -m aopd export --checkpoint /path/to/global_step_N --output /path/to/new-hf-model
```

Only export trusted checkpoints: the upstream FSDP merger uses Python pickle.
Resume requires the original optimizer and dataloader/task-state files and an
unchanged scientific protocol. Use a new run directory when extending a run;
do not change task quotas, splits, teacher settings or source data on resume.
FSDP resume also requires the original training GPU count (GPU IDs may change).
The launcher checks every rank's model/optimizer/extra-state files; it does not
claim to repair corrupted pickle files or reshard a checkpoint. `--smoke` is
fresh-run only. The public launcher ignores inherited experimental shell
variables; set experiment changes in YAML, not legacy `SMOKE`/`TAU_*` exports.

## Periodic validation

Validation is off by default (`every_steps: -1`, `before_train: false`). Enable it
in the runtime YAML:

```yaml
validation:
  every_steps: 10
  before_train: true
  domains: [airline, retail, telecom]
  split: test
  trials: 1
  batch_size: 16
  max_steps: 30
```

Tau defaults to its training user simulator and the three official test domains.
Set `roles.validation_user` to a named service for an independent simulator,
with the same generation controls as `roles.user`. Main training defaults to
Airline/base and requires explicit `sources.tau` and `roles.validation_user`
when validation is enabled; it does not reuse the EnvScaler user implicitly.
For example, a configured auxiliary service can also serve validation:

```yaml
roles:
  # Keep the other training roles in this mapping.
  validation_user:
    service: auxiliary
    generation: {enable_thinking: false, temperature: 1.0, max_tokens: 8192}
```

Validation reuses the student rollout engine with greedy decoding, without
teacher or matcher queries. Fixed domain quotas produce complete batches;
`data/manifest.json` (Tau) or `data/tau_validation/manifest.json` (main) records
the evaluated and dropped tail rows. Disabled validation and bounded smoke do
not start a dedicated validation service or environment. Export checkpoints
and use external runners for final benchmark evaluation.

## Tau outcome-GRPO

This separate shell entry point needs only a user-simulator endpoint; it does
not read runtime YAML or start model services. Set `STUDENT_MODEL`,
`TRAIN_PYTHON`, `TAU_SOURCE`, `AUX_MODEL`, `AUX_BASE_URL` and `AUX_API_KEY` as
above, then run:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
  PYTHON="$TRAIN_PYTHON" MODEL_PATH="$STUDENT_MODEL" TAU2_ROOT="$TAU_SOURCE" \
  METHOD=outcome N_GPUS=2 TP_SIZE=1 SP_SIZE=2 \
  TRAIN_STEPS=50 AIRLINE_TRAJ=8 RETAIL_TRAJ=8 TELECOM_TRAJ=8 \
  TAU_USER_MODEL="openai/$AUX_MODEL" TAU_USER_API_BASE="$AUX_BASE_URL" \
  TAU_USER_API_KEY_ENV=AUX_API_KEY TAU_USER_REASONING_ENABLED=false \
  TAU_USER_TEMPERATURE=0.7 TAU_USER_TOP_P=0.8 \
  TAU_USER_TOP_K=20 TAU_USER_PRESENCE_PENALTY=0 \
  RUN_DIR="runs/$(date -u +%Y%m%dT%H%M%SZ)-tau-outcome" \
  bash examples/tau_bench/train/run.sh
```

For eight training GPUs set `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`, `N_GPUS=8`,
`TP_SIZE=2`, `SP_SIZE=4`. Add `SMOKE=1 SMOKE_AIRLINE_TRAJ=2 SMOKE_RETAIL_TRAJ=0`
before `bash` for a two-GPU one-step smoke; set `SMOKE_AIRLINE_TRAJ=8` for eight GPUs.
Use only GPUs not assigned to other jobs or model services. The default
trajectory count per task is four (`ROLLOUT_N=4`). This baseline uses outcome
rewards, not the AOPD reward/advantage protocol above.
For a vLLM endpoint, keep the `openai/` prefix in `TAU_USER_MODEL` for LiteLLM
routing; it is not part of the server's model name.
