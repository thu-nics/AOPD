# AgentWorldModel integration

This integration exposes two deliberately separate protocols:

- `awm_semantic`: state-group semantic action distillation with four student
  candidates and three independent DeepSeek teacher samples per visited state.
- `awm_outcome`: ordinary four-trajectory GRPO with only AWM's terminal verifier
  reward. It does not create a teacher or semantic matcher.

It targets the complete public `Snowflake/AgentWorldModel-1K` revision
`dde80a0283fe781bdc51656bce57063dc5650213`: 1,000 environments, ten tasks per
environment, and 10,000 tasks total. The 526-environment/3,315-task numbers in
the paper describe a compute-limited training subset and are not treated as the
public dataset cardinality.

## Fixed protocol

- OpenEnv commit: `5298e0d91c6cd55d5f3a81259d5b2a9a1e05eff0`.
- Student context: 32,000 tokens total, split into 27,904 prompt tokens and
  4,096 response tokens.
- The system message and task are pinned. At most six complete recent
  action/result exchanges are retained. The exact budget-trimmed chat and the
  same actual environment-tool schemas are shared by all four candidates and
  the teacher.
- There is no model-visible `list_tools` scaffold or nested `call_tool` meta
  tool. The adapter fetches schemas internally after reset, then Qwen3 and
  DeepSeek each receive those actual tools through their native function-calling
  interface. The adapter deterministically removes AWM's contradictory sibling
  `type: T` when the same node already declares `anyOf: [T, null]`; raw and
  canonical schema hashes plus every repair remain audit-visible. Duplicate
  names in JSON Schema `required` arrays are also losslessly deduplicated while
  preserving first-occurrence order. Because AWM's server still validates calls
  against the contradictory raw schema, explicit `null` on non-required nullable
  fields is canonicalized to argument omission before semantic matching and
  execution (equivalent to the upstream Python `Optional[T] = None` default).
- Every state obtains an ordered K=3 teacher multiset. Duplicate actions are
  retained. Tool calls match by canonical tool name and exact canonical
  arguments. Message/final actions use normalized exact match and then one
  frozen pairwise semantic judgment per candidate/teacher pair.
- Candidate rewards are the number of matching teacher samples, legal unmatched
  actions receive zero, and invalid actions receive -1. Only a uniform choice
  among maximum-reward candidates executes.
- Teacher or matcher failure masks the complete group; it is never converted to
  a false/non-match label. Equal-reward groups are also masked.
- A selected ordinary message is a terminal communicative action. Every
  successfully reset episode is finalized by AWM's official SQL plus
  code-augmented LLM judge. Its result is logged for semantic training but never
  enters semantic reward, advantage, group selection, or loss masking.
- Training and internal evaluation retain the same configurable action-exchange
  history, defaulting to the six most recent exchanges, and a 20-decision
  action budget. The healthy training pool uses the same native prompt and
  16K fixed-scaffold cutoff, strict scenario/SQL-verifier checks, and a no-action
  SQL+LLM health audit. Expert task success is not a membership gate.
- Strong runtime infrastructure failures directly mask and end only the affected
  state group. They are logged run-locally without replay or a persistent task
  quarantine.

The direct DeepSeek API model ID is `deepseek-v4-flash`. Teacher calls enable
thinking with `reasoning_effort=max`. DeepSeek ignores `temperature` and `top_p`
in thinking mode, so the teacher protocol does not send or cache those ineffective
controls. Matcher calls disable thinking and use temperature zero. Training metrics
separate live API requests/tokens from cache hits and loaded-cache inventory, so a
resumed run does not report historical cache cost as new usage. The API key is read
from `DEEPSEEK_API_KEY` and is never written to a dataset, manifest, or cache.
Each cached response records the provider-returned model and system fingerprint;
an identity change within one cache/run fails instead of mixing teacher versions.

Semantic DAPO reports `dapo/awm/oracle_hit_rate`, the fraction of supervised,
non-padding candidate actions that match at least one action in the teacher
multiset. Equal-reward groups remain in this diagnostic even though they do not
produce gradients; teacher, matcher, and runtime-infrastructure masked rows do
not enter its denominator. It also reports `dapo/awm/skipped_oracle_rate`, the
fraction of candidate rows in fully skipped state groups that match at least one
teacher action, and `dapo/awm/skipped_all_oracle_group_rate`, the fraction of
fully skipped state groups whose every candidate matches the teacher multiset.

## Install and data

The existing development environment can be reused:

```bash
export PYTHON=/opt/venvs/verl-agent/bin/python
bash examples/awm/setup/install_awm.sh
```

Local development defaults are centralized in `common/paths.sh`: `VENV_PATH`,
`AWM_SOURCE_DIR`, and `AWM_CACHE_DIR` point to the shared installation under
`/opt/venvs` and `/mnt/public2/yuanhuining/repos`. Existing `PYTHON`,
`OPENENV_ROOT`, and `AWM_DATA_DIR` overrides remain supported. Reusable AWM
logic lives in `agent_system.environments.env_package.awm`;
`examples/awm/{setup,runtime,data,screening,train,eval}` separates installation,
server lifecycle, data processing, screening, training, and standalone eval.

The installer refuses to mutate an existing OpenEnv checkout at another commit.
Network commands honor the standard proxy variables from the shell.

Prepare the complete pinned dataset and deterministic split manifest:

```bash
$PYTHON examples/awm/data/prepare_data.py \
  --data-dir "/mnt/public2/yuanhuining/repos/openenv-awm-cache" \
  --output-dir data/awm
```

This validates 1,000×10 task cardinality, uniqueness, and pinned source
identity, then writes:

- `awm_all.parquet`: 1,000 environments / 10,000 tasks; and
- `manifest.json`: source hashes, revision, selection rule, counts, and the
  historical logical `all`/`dev`/`smoke` task IDs.

Only the complete Parquet is materialized. Development, smoke, and evaluation
slices use a deterministic verified-pool selection or an explicit task limit.

Dev and smoke selection uses only SHA-256 ranks of scenario/task IDs. It never
uses expert output, verifier outcome, or student performance.

## Build the verifier-reliable healthy task pool

The formal data path has one intentionally small membership rule:

```text
10,000 pinned public tasks
  -> 9,380 native-tool fixed prompts <= 16K
  -> healthy iff scenario database build, SQL verifier, and no-action SQL+LLM audit pass
```

First materialize the hash-bound context selection:

```bash
bash examples/awm/data/run_selection.sh
```

Then run the resumable health audit:

```bash
bash examples/awm/data/run_healthy_pool.sh
```

The audit has only `healthy` and `quarantine` outcomes; there is no pending or
manual-review class. It checks:

- the pinned source hashes and exact 9,380-task candidate order;
- each scenario's unique task/schema/sample records and ten-task cardinality;
- a fresh SQLite build in which every table DDL, index, and seed INSERT must
  succeed (one failure quarantines the whole scenario);
- one unique SQL verifier matching the exact task text, compiling successfully,
  and defining `verify_task`; the pure-code verifier is ignored; and
- a fresh-reset, no-action invocation of AWM's official SQL verifier plus its
  code-augmented DeepSeek judge.

A no-action `incomplete` or `agent_error` result is healthy. No-action
`complete` or `server_error` is quarantined. Judge/API/timeout failures receive
three fresh-reset attempts and are conservatively quarantined if exhausted.
This stage calls the API once per healthy deterministic task in the common
case, so its cost is explicit rather than hidden inside training.

Existing `runs/awm_final_pool/trials.jsonl` is read by default only to attach a
compact one-off expert outcome to each row. Expert success or failure never
changes membership. Set `EXPERT_TRIALS=` to omit that metadata. The immutable,
hash-bound output is:

- `runs/awm_healthy_pool/awm_training_pool.parquet`;
- `runs/awm_healthy_pool/health_manifest.json`;
- `runs/awm_healthy_pool/scenario_health.jsonl`; and
- `runs/awm_healthy_pool/task_health.jsonl`.

The AWM server used by preprocessing retains the official SQL evidence builder,
official judge prompt, and official label parser. A repository-owned transport
layer routes that judge to `deepseek-v4-flash`, enables native thinking with
`reasoning_effort=max`, allows 8,192 response tokens, and retries judge errors.
The health audit itself uses exactly three fresh resets; training terminal
judging defaults to five attempts.

During semantic training, schema-valid tool HTTP 5xx responses still use the
separate code-augmented runtime-error judge. High-confidence
`policy_execution_error` actions receive reward `-1`; unchanged states may
continue. Strong infrastructure failures terminate and mask only the affected
state group. This runtime protection neither replays trajectories nor creates a
persistent task blacklist.

Train the healthy pool with:

```bash
TRAIN_DATA=runs/awm_healthy_pool/awm_training_pool.parquet \
TRAIN_SELECTION_MANIFEST=runs/awm_healthy_pool/health_manifest.json \
MODEL_PATH=/mnt/public2/yuanhuining/models/Qwen3-4B \
  bash examples/awm/train/run_semantic.sh
```

The launcher verifies all manifest and artifact hashes before optional slicing
and schedule materialization. Without explicit `TRAIN_DATA`, this healthy pool
is the formal-training default. `USE_RAW_SPLIT=1` is diagnostic-only.

## Start AWM for preprocessing

AWM scenario code is trusted research code and is not isolated inside the server
process beyond the outer machine/container boundary. Run it only on a suitable
development host.

```bash
export AWM_SOURCE_DIR=/mnt/public2/yuanhuining/repos/openenv-awm
export AWM_CACHE_DIR="/mnt/public2/yuanhuining/repos/openenv-awm-cache"
bash examples/awm/runtime/start_server.sh
```

Context selection and deterministic integrity preprocessing expect this
standalone server at `http://127.0.0.1:8000`. Training does not reuse it.

## Train

Semantic training requires `DEEPSEEK_API_KEY` for the teacher. The Tau
user-simulator provider is selected by `TAU_USER_LLM`; its default OpenRouter
model requires `OPENROUTER_API_KEY`. To route every external request through
the official DeepSeek API, set
`TAU_USER_LLM=deepseek/deepseek-v4-flash`:

The Tau adapter keeps user-simulator reasoning disabled with provider-native
arguments: official DeepSeek uses `thinking.type=disabled`, while OpenRouter
models retain `reasoning.enabled=false`.

```bash
MODEL_PATH=/mnt/public2/yuanhuining/models/Qwen3-4B \
  bash examples/awm/train/run_semantic.sh
```

Each training run starts a dedicated AWM server on an automatically selected
localhost port. The launcher verifies both the pinned logical-time protocol and
a per-run server identity before constructing environments. Server output and
its protocol manifest are written to `awm_server.log` and
`awm_server_manifest.json` inside the run directory. Normal completion,
failure, `SIGINT`, and `SIGTERM` stop the complete server process group so that
scenario subprocesses cannot leak across runs. `AWM_PORT` requests a specific
free port. Reusing an explicitly managed external service is an opt-out for
diagnostics only: set `MANAGE_AWM_SERVER=0` together with `AWM_BASE_URL`.

For a non-smoke semantic run, the launcher defaults to the verified healthy
pool under `runs/awm_healthy_pool`; it no longer uses `TRAIN_SPLIT=all`
implicitly. The pool is independent of one-off expert success. Formal defaults are 200 optimizer steps, 64 tasks per step, four
student candidates per state, two A800 GPUs, save every 10 steps, and validation
at step 0 and every 20 steps. Checkpoints are retained without a default cap.
Set `USE_RAW_SPLIT=1` only for an explicit diagnostic run. Every formal pool
launch verifies the source hashes, manifest, Parquet hash, and ordered task IDs.

`TRAIN_TASK_FRACTION` or `TRAIN_TASK_COUNT` selects a reproducible ordered
prefix of the healthy task pool. Fractions are defined against the 9,380
context-eligible tasks, not against the active-pool size, so `0.1` selects 938
tasks when that many remain. The source is environment-round-robin ordered; filtering preserves that order
without duplicating tasks. It is therefore a deterministic 10%
task slice, not an exact one-task-per-environment slice:

```bash
TRAIN_TASK_FRACTION=0.1 MODEL_PATH=/mnt/public2/yuanhuining/models/Qwen3-4B \
  bash examples/awm/train/run_semantic.sh
```

The run stores and hash-verifies its slice Parquet and manifest under
`runs/<UTC timestamp>/data/`. The two controls are mutually exclusive.

One-step development smoke, including two official Airline validation tasks:

```bash
MODEL_PATH=/mnt/public2/yuanhuining/models/Qwen3-4B \
  bash examples/awm/train/run_semantic_smoke.sh
```

The isolated outcome baseline uses DeepSeek only for the shared terminal SQL+LLM judge:

```bash
MODEL_PATH=/mnt/public2/yuanhuining/models/Qwen3-4B \
  bash examples/awm/train/run_outcome.sh
```

All machine-specific paths, batch sizes, GPU settings, run directories, and
managed-server host/port settings are environment-variable overrides in
`run_training.sh`. By default,
each launch writes under `runs/<UTC timestamp>/`; TensorBoard event files live in
that run's `tensorboard/` subdirectory instead of a repository-level
`tensorboard_log/`. `RUN_DIR` and `TENSORBOARD_DIR` remain explicit overrides.
The default 32,000-token student context reserves 4,096 tokens for each generated
action and 27,904 tokens for its prompt. Override `MAX_RESPONSE_LENGTH` or
`MAX_MODEL_LEN` as needed; when `MAX_PROMPT_LENGTH` is unset, the launcher derives
it as `MAX_MODEL_LEN - MAX_RESPONSE_LENGTH` and rejects inconsistent explicit
budgets. `MAX_NUM_BATCHED_TOKENS` defaults to `MAX_MODEL_LEN`.
Training and standalone AWM evaluation keep the complete logical interaction
history. At render time they pin the system message, initial task, and complete
native tool schemas, then drop the oldest whole action/observation exchanges
until the prompt fits `MAX_PROMPT_LENGTH`; a half tool exchange is never kept.
`MAX_HISTORY_EXCHANGES` is empty by default and may be set to a non-negative
integer only for a fixed-window ablation. `max_steps` remains independently
fixed at 20 for the main AWM protocol. This context-policy change bumps AWM
runtime, oracle-cache, and native-eval protocol identities, so old caches and
evaluation artifacts are not resume-compatible.
Each new semantic run evaluates fixed-composition, complete Tau validation
batches at step 0 and every 20 steps with the resident training vLLM. The agent
uses greedy decoding (`do_sample=false`, `temperature=0`), while the DeepSeek
user simulator remains stochastic at temperature 1 with thinking disabled.
Training rollout sampling remains `temperature=0.6`, `top_p=0.95`, and
`top_k=20`. With the default `VAL_BATCH=16`, Airline-only evaluates 48 of 50
official `base` tasks. Set
`TAU_VAL_DOMAINS=airline,retail` for a fixed 5-Airline/11-Retail template that
evaluates 50 plus 110 tasks, or `VAL_BEFORE_TRAIN=false` only when intentionally
skipping the baseline. Validation uses seed 300, one trial per task, and no
teacher; its manifest records evaluated and dropped rows. Full official-split
evaluation is a separate manual native-runner step, never an automatic training
finalizer. On save/validation overlaps, the checkpoint is written first.
The two-GPU default uses `SP_SIZE=2`. Both AWM variants use the paper setting
`entropy_coeff=0`; they disable the otherwise metric-only full-vocabulary
entropy recomputation, which is not part of the loss and is prohibitively large
at this context length. Low-memory monitoring is enabled instead. It reuses the
selected-token log probabilities to emit `rollout/sampled_token_entropy_all`
for every real generated token and `actor/sampled_token_entropy_train` for the
tokens retained after state-group masking. Because top-k/top-p sampling is used,
these are sampled-surprisal entropy proxies rather than exact distribution
entropy. Canonical tool-use collapse is monitored independently through
`state_group/awm/canonical_unique_action_count_mean`,
`state_group/awm/canonical_unique_action_rate`, and
`state_group/awm/canonical_all_identical_rate`. None of these metrics changes the
loss. The actor dynamic-microbatch default is 16,384 tokens
per GPU; with sequence parallel size 2 this still admits one complete 32,000-token
sequence while avoiding the near-capacity peak caused by batching two of them.

## Native standalone evaluation

`eval/eval_awm.py` uses `AWMEnv` and its native `reset`, `step`, `verify`, and
`done` methods directly. It fetches schemas internally once after reset and
passes the actual tools through Qwen3's native tool template; `list_tools` and
`call_tool` are not model actions. The vLLM launcher uses the `hermes` parser
matching this checkpoint's JSON-in-`<tool_call>` template, the `qwen3`
reasoning parser, and Qwen3's recommended thinking-mode sampling
(`temperature=0.6`, `top_p=0.95`, `top_k=20`) with the recorded evaluation
seed. It never creates the training Ray rollout stack. `eval/run_eval.sh`
starts one OpenAI-compatible vLLM server, keeps the model
resident while every selected task runs, and stops only that server process at
the end. Native eval defaults to the official recommended `sql` verifier:
OpenEnv builds SQL/code evidence and applies its official LLM-judge prompt,
while the pinned AWM server supplies the repository's DeepSeek thinking-mode
transport. Set `DEEPSEEK_API_KEY` before launching:

```bash
SPLIT=all TASK_LIMIT=8 MODEL_PATH=/mnt/public2/yuanhuining/models/Qwen3-4B \
  bash examples/awm/eval/run_eval.sh
```

Set `START_VLLM=0 API_BASE=http://host:port/v1` to use an already persistent
server. `VERIFIER_MODE=code` remains available only as a pure-code verifier
ablation. `JUDGE_MODEL`, `JUDGE_API_BASE`, and `JUDGE_API_KEY_ENV` configure the
SQL judge without placing the API key in command-line arguments or artifacts.
The launcher rejects an AWM server whose reported judge model, reasoning mode,
or response budget differs from the requested protocol. `SEED` defaults to 300
and is part of every result and the strict run identity. Evaluation
incrementally writes raw per-task trajectories and verifier results before
producing a summary. `--resume` is accepted only when the complete protocol
identity matches, including the non-secret terminal-judge server protocol.
For local checkpoints, that identity hashes the contents of every artifact file,
including all weight shards.

To inspect Qwen3-4B function-call parsing on the expert-qualified diagnostic
tasks without loading/offloading the training rollout engine, use the native
evaluation process:

```bash
DATA_FILE=runs/awm_healthy_pool/awm_training_pool.parquet \
SELECTION_MANIFEST=runs/awm_healthy_pool/health_manifest.json \
MODEL_PATH=/mnt/public2/yuanhuining/models/Qwen3-4B \
TASK_LIMIT=32 SPLIT=all bash examples/awm/eval/run_eval.sh
```

Its summary reports verifier success both over all tasks and over valid terminal
judge outcomes, terminal label coverage/counts, action-kind counts, parse
failures, schema-valid tool calls, tool execution errors, decision counts, and
token use;
the JSONL retains every raw action and parsed action for manual inspection.
