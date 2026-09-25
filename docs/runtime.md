# Runtime configuration

Science and placement are separate. A recipe fixes the experiment; a runtime
YAML names models, endpoints, paths and devices. `${NAME}` requires that
environment variable; missing variables fail immediately. No private cluster
paths, IPs or secret values are supplied by the repository.

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

Self-AOPD uses only the user and matcher roles; an unused external teacher is
not started or contacted. Unsupported
generation keys fail explicitly rather than being silently ignored.

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
