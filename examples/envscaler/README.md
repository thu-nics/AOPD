# EnvScaler integration

This integration adds EnvScaler conversation tasks to AWM semantic-action
training without changing upstream `verl` behavior. It is pinned to EnvScaler
commit `87e667397abacf274858c0964796beb8f984aafe` and validates hashes for all
three metadata files before constructing an environment. The pinned RL split
contains 2,550 tasks: 50 tasks for each of 51 environments. The complete source
repository remains at `/mnt/public2/yuanhuining/repos/EnvScaler`.

## Runtime protocol

- Student and teacher see the same real environment functions as native tools;
  there is no `list_tools` action or nested `call_tool` wrapper.
- Every state samples four student candidates and a three-sample ordered teacher
  multiset. Tool, ordinary-message, and invalid candidates use the existing AWM
  frequency-weighted semantic reward and exactly one uniform-argmax candidate
  advances the environment.
- A DeepSeek user simulator starts each conversation and responds to ordinary
  assistant messages. It is fixed to temperature 1 with DeepSeek-native
  `thinking={"type":"disabled"}`, matching the Tau user-simulator protocol;
  no `top_p` or output-token override is sent. `###STOP###` is
  accepted only after the deterministic checkers are complete; an early stop is
  repaired once and a repeated early stop is an infrastructure failure.
- Tool exceptions restore the exact pre-call object state and return a local
  error observation. The trajectory can continue and learn from that error.
- EnvScaler trajectories allow at most 40 student decisions. AWM trajectories
  in the same batch retain their independent 20-decision limit.
- The agent chat keeps the full logical history. Rendering always pins the
  system message, initial user request, and complete tool schemas, then retains
  as many newest whole exchanges as fit under `data.max_prompt_length`. The main
  setting uses 27,904 prompt tokens plus 4,096 response tokens. An optional
  `MAX_HISTORY_EXCHANGES` cap exists only for ablations and is empty by default.

## Health filter

The filter deliberately has two simple stages:

1. Deterministic audit: quarantine tasks with missing/empty checkers, more than
   100 checkers, exact duplicate checker code, source/class/init/tool-schema or
   method-contract failures, non-reproducible fresh reset, checker execution
   errors, or an initially complete state.
2. Static code-augmented DeepSeek audit: provide the task, authoritative
   post-`init_config` runtime state, native initialization semantics, environment
   code, native tools, checker code, and no-action checker results to a
   task-scoped judge modeled on AWM's code-augmented protocol. A legal workaround
   is accepted when it preserves the task constraints; unused broken tools and
   unrelated schema/documentation defects do not reject a task. Checkers that
   cannot distinguish a requested change from an incorrect or no-action state
   still make the task unreliable. `healthy` is accepted;
   `environment_or_verifier_failure`, `uncertain`, and exhausted API failures
   are quarantined. Confidence is diagnostic only. No expert trajectory is
   generated.

The numbered processing stages are `01_deterministic_audit` followed by
`02_code_augmented_screening`. The current full deterministic audit is under
`runs/envscaler_data_processing/01_deterministic_audit`: 2,495 pass and 55 quarantine (53 exact
duplicate-checker tasks, one 445-checker task, and one checker runtime failure).
The completed code-augmented stage classifies 738 tasks from 47 environments as
healthy and quarantines the other 1,757 deterministic-pass tasks; there are no
pending infrastructure records. The final durable funnel is therefore
2,550 source tasks -> 2,495 deterministic pass -> 738 judge-healthy tasks.
Both stages remain resumable.

Run a small paid smoke before a full screen:

```bash
export DEEPSEEK_API_KEY=...
PYTHON=/opt/venvs/verl-agent/bin/python \
OUTPUT_DIR=/tmp/envscaler_filter_smoke \
DETERMINISTIC_DIR=runs/envscaler_data_processing/01_deterministic_audit \
LIMIT=4 CONCURRENCY=1 \
bash examples/envscaler/filter/run_full_filter.sh
```

Run or resume the full filter only after reviewing smoke cost:

```bash
export DEEPSEEK_API_KEY=...
PYTHON=/opt/venvs/verl-agent/bin/python \
OUTPUT_DIR=runs/envscaler_data_processing/02_code_augmented_screening \
CONCURRENCY=16 RESUME=0 \
bash examples/envscaler/filter/run_full_filter.sh

# After an interruption; concurrency may be changed safely.
PYTHON=/opt/venvs/verl-agent/bin/python \
OUTPUT_DIR=runs/envscaler_data_processing/02_code_augmented_screening \
CONCURRENCY=8 RESUME=1 \
bash examples/envscaler/filter/run_full_filter.sh
```

The durable outputs are `config.json`, `task_audit.jsonl`,
`health_manifest.json`, and `envscaler_training_pool.parquet`. Resume verifies
the source commit, metadata hashes, task count, model, judge protocol, exact
eligible task IDs, and agreement between every deterministic JSONL record and
its manifest. The health manifest records the complete two-stage funnel. The
reviewed v2-to-v3 migration reuses legacy `healthy`
records and unambiguous task-local failures, while selectively refreshing
infrastructure failures and verdicts whose rationale relied on constructor-only
initialization or broad, non-task-scoped contract defects. Replaced reviews are
retained in `health_review_history`. API token totals are reconstructed from
current and historical durable judge records.

The DeepSeek screening judge uses a 32,768-token response budget by default so
max-effort thinking does not consume the entire budget before emitting the JSON
verdict. Resume accepts an otherwise-identical v2 or v3 configuration with the
earlier 8,192- or 16,384-token budget and retries only records still marked as
infrastructure failures.

## Mixed semantic training

The formal mixed launcher creates a deterministic 64-task schedule containing
exactly 48 AWM and 16 EnvScaler trajectories per RL step. Each family is sampled
round-robin by environment before a task is reused. Periodic validation remains
the existing fixed-domain Tau validation; there is no automatic final full eval.

```bash
export DEEPSEEK_API_KEY=...
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
PYTHON=/opt/venvs/verl-agent/bin/python \
MODEL_PATH=/mnt/public2/yuanhuining/models/Qwen3-8B \
TRAIN_DATA=runs/awm_data_processing/03_code_augmented_screening/awm_training_pool.parquet \
TRAIN_SELECTION_MANIFEST=runs/awm_data_processing/03_code_augmented_screening/health_manifest.json \
ENVSCALER_POOL=runs/envscaler_data_processing/02_code_augmented_screening/envscaler_training_pool.parquet \
ENVSCALER_MANIFEST=runs/envscaler_data_processing/02_code_augmented_screening/health_manifest.json \
TAU_USER_LLM=deepseek/deepseek-v4-flash \
N_GPUS=8 TP_SIZE=2 SP_SIZE=4 \
PPO_MAX_TOKENS_PER_GPU=8192 LOGPROB_MAX_TOKENS_PER_GPU=8192 \
TRAIN_STEPS=200 TRAIN_BATCH=64 \
AWM_PER_STEP=48 ENVSCALER_PER_STEP=16 \
SAVE_FREQ=10 TEST_FREQ=20 \
bash examples/envscaler/train/run_mixed_semantic.sh \
  env.awm.oracle.model=deepseek-v4-flash \
  env.awm.oracle.api_key_env=DEEPSEEK_API_KEY
```

Dynamic batching still sees the padded 32,000-token sequence dimension. Keep
both `PPO_MAX_TOKENS_PER_GPU * SP_SIZE` and
`LOGPROB_MAX_TOKENS_PER_GPU * SP_SIZE` at least `MAX_MODEL_LEN`; the launcher
fails before rollout if this invariant is violated. The 8-GPU recipe above uses
SP=4 and 8,192 tokens per GPU, giving 32,768 tokens of logical capacity.

Use `examples/envscaler/train/run_mixed_semantic_smoke.sh` for a one-step GPU
smoke after the health pool exists. Runtime code lives under
`agent_system/environments/env_package/envscaler/`; this examples directory
contains only data, filter, and launcher entry points.
