# EnvScaler integration

This integration adds EnvScaler conversation tasks to AWM Agentic OPD
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
  frequency-weighted semantic reward. Exactly one maximum-reward candidate
  advances the environment; tied maxima prefer a canonical action different
  from the immediately previous committed action.
- A DeepSeek user simulator starts each conversation and responds to ordinary
  assistant messages. It is fixed to temperature 1 with DeepSeek-native
  `thinking={"type":"disabled"}`, matching the Tau user-simulator protocol;
  no `top_p` or output-token override is sent. `###STOP###` is a natural
  conversation terminal signal even when the deterministic checkers remain
  incomplete. Checkers still record terminal success and partial completion,
  but never force a stopped conversation to continue.
- A schema-valid tool exception restores the exact pre-call object state and
  invokes the cached, code-augmented DeepSeek runtime judge. A high-confidence
  policy execution error assigns `-1` to every identical canonical candidate
  and continues from the restored state. Infrastructure, uncertain,
  low-confidence, and judge-failure cases mask the current state group and end
  the trajectory. Normal tool calls never invoke this judge.
- After two identical tool calls return the same observation, a prospective
  third identical call has positive semantic reward capped at zero. Tied
  maximum-reward commits prefer a different canonical action, and an actual
  fourth no-progress repeat ends the trajectory. There is no dynamic
  resampling.
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
2. Static feasibility judge (code-augmented): provide the task, authoritative
   post-`init_config` runtime state, native initialization semantics, environment
   code, native tools, checker code, and no-action checker results to a
   task-scoped judge modeled on AWM's code-augmented protocol. A legal workaround
   is accepted when it preserves the task constraints; unused broken tools and
   unrelated schema/documentation defects do not reject a task. Checkers that
   cannot distinguish a requested change from an incorrect or no-action state
   still make the task unreliable. `healthy` is accepted;
   `environment_or_verifier_failure` and `uncertain` are quarantined.
   Exhausted API failures remain `pending` and must succeed on a later
   resume before the healthy manifest is built. Confidence is diagnostic only.
   No expert trajectory is generated.

The numbered processing stages are `01_deterministic_audit` followed by
`02_static_feasibility_judge`. The current full deterministic audit is under
`runs/envscaler_data_processing/01_deterministic_audit`: 2,495 pass and 55 quarantine (53 exact
duplicate-checker tasks, one 445-checker task, and one checker runtime failure).
The existing artifact contains 1,071 reviews that already match the sole current
protocol and 1,424 stale reviews. On the next resume, the 1,071 matching reviews
are reused and every stale review is refreshed. The final healthy count may therefore
change. Until that refresh completes and rewrites the manifest, mixed-training
launches intentionally reject the stale manifest. Both processing stages remain
resumable.

Run a small paid smoke before a full screen:

```bash
export DEEPSEEK_API_KEY=...
PYTHON=/opt/venvs/verl-agent/bin/python \
OUTPUT_DIR=/tmp/envscaler_filter_smoke \
DETERMINISTIC_DIR=runs/envscaler_data_processing/01_deterministic_audit \
LIMIT=4 CONCURRENCY=1 \
bash examples/envscaler/data/run_static_feasibility_judge.sh
```

Run or resume the full filter only after reviewing smoke cost:

```bash
export DEEPSEEK_API_KEY=...
PYTHON=/opt/venvs/verl-agent/bin/python \
OUTPUT_DIR=runs/envscaler_data_processing/02_static_feasibility_judge \
CONCURRENCY=16 RESUME=auto \
bash examples/envscaler/data/run_static_feasibility_judge.sh

# After an interruption; concurrency may be changed safely.
PYTHON=/opt/venvs/verl-agent/bin/python \
OUTPUT_DIR=runs/envscaler_data_processing/02_static_feasibility_judge \
CONCURRENCY=8 RESUME=1 \
bash examples/envscaler/data/run_static_feasibility_judge.sh
```

The durable outputs are `config.json`, `task_audit.jsonl`,
`health_manifest.json`, and `envscaler_training_pool.parquet`. Resume verifies
the source commit, metadata hashes, task count, model, judge protocol, exact
eligible task IDs, and agreement between every deterministic JSONL record and
its manifest. The health manifest records the complete two-stage funnel. All
judge verdicts must use the sole shared static-feasibility protocol. Protocol-mismatched
reviews and exhausted infrastructure attempts are refreshed; exact current-protocol
reviews are reused. Replaced reviews are retained in `health_review_history`.
API token totals are reconstructed from current and historical durable records.

The DeepSeek screening judge uses a 32,768-token response budget by default so
max-effort thinking does not consume the entire budget before emitting the JSON
verdict. Its temperature, thinking mode, reasoning effort, and token budget are
bound into the resumable configuration. AWM and EnvScaler use the same generation
contract. Environments run in parallel, while tasks from one environment run in
source order to improve prefix-cache reuse. `TIMEOUT_SECONDS`, `MAX_RETRIES`, and
`MAX_TOKENS` remain explicitly configurable.

## Mixed Agentic OPD training

The formal mixed launcher creates a deterministic 64-task schedule containing
exactly 58 AWM and 6 EnvScaler trajectories per RL step. Each family is sampled
round-robin by environment before a task is reused. Periodic validation remains
the existing fixed-domain Tau validation; there is no automatic final full eval.

```bash
export DEEPSEEK_API_KEY=...
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
PYTHON=/opt/venvs/verl-agent/bin/python \
MODEL_PATH=/mnt/public2/yuanhuining/models/Qwen3-8B \
TRAIN_DATA=runs/awm_data_processing/03_static_feasibility_judge/awm_training_pool.parquet \
TRAIN_SELECTION_MANIFEST=runs/awm_data_processing/03_static_feasibility_judge/health_manifest.json \
ENVSCALER_POOL=runs/envscaler_data_processing/02_static_feasibility_judge/envscaler_training_pool.parquet \
ENVSCALER_MANIFEST=runs/envscaler_data_processing/02_static_feasibility_judge/health_manifest.json \
TAU_USER_LLM=deepseek/deepseek-v4-flash \
N_GPUS=8 TP_SIZE=2 SP_SIZE=4 \
PPO_MAX_TOKENS_PER_GPU=8192 LOGPROB_MAX_TOKENS_PER_GPU=8192 \
TRAIN_STEPS=200 TRAIN_BATCH=64 \
AWM_PER_STEP=58 ENVSCALER_PER_STEP=6 \
SAVE_FREQ=10 TEST_FREQ=20 \
bash examples/agentic_opd/run_mixed_agentic_opd.sh \
  env.awm.oracle.model=deepseek-v4-flash \
  env.awm.oracle.api_key_env=DEEPSEEK_API_KEY
```

Dynamic batching still sees the padded 32,000-token sequence dimension. Keep
both `PPO_MAX_TOKENS_PER_GPU * SP_SIZE` and
`LOGPROB_MAX_TOKENS_PER_GPU * SP_SIZE` at least `MAX_MODEL_LEN`; the launcher
fails before rollout if this invariant is violated. The 8-GPU recipe above uses
SP=4 and 8,192 tokens per GPU, giving 32,768 tokens of logical capacity.

Use `examples/agentic_opd/run_mixed_agentic_opd_smoke.sh` for a one-step GPU
smoke after the health pool exists. Runtime code lives under
`agent_system/environments/env_package/envscaler/`; the EnvScaler examples
directory exposes data preparation and health-filter entry points under
`examples/envscaler/data/`. The former `examples/envscaler/filter/*.sh`
entry points remain as compatibility redirects; they contain no filtering
implementation. Cross-environment
training launchers live under `examples/agentic_opd/`.
