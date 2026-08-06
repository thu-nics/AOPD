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
- A selected ordinary message is a terminal communicative action. The code
  verifier runs only for outcome reporting; its result is not added to semantic
  training reward.
- Training and internal evaluation retain the same configurable action-exchange
  history, defaulting to the six most recent exchanges, and a 20-decision
  action budget. The deterministic training pool uses the same native prompt
  and 16K fixed-scaffold cutoff but no expert-success qualification gate.
- Strong runtime environment failures are replayed from a fresh reset with the
  exact structured tool-call prefix and no model calls. Confirmed defects mask
  the whole reset and enter the run-local quarantine; transient infrastructure
  failures mask only the affected trajectory.

The direct DeepSeek API model ID is `deepseek-v4-flash`. Teacher calls enable
thinking with `reasoning_effort=max`. DeepSeek ignores `temperature` and `top_p`
in thinking mode, so the teacher protocol does not send or cache those ineffective
controls. Matcher calls disable thinking and use temperature zero. Training metrics
separate live API requests/tokens from cache hits and loaded-cache inventory, so a
resumed run does not report historical cache cost as new usage. The API key is read
from `DEEPSEEK_API_KEY` and is never written to a dataset, manifest, or cache.
Each cached response records the provider-returned model and system fingerprint;
an identity change within one cache/run fails instead of mixing teacher versions.

Semantic DAPO also reports `dapo/awm/skipped_oracle_rate`, the fraction of
candidate rows in fully skipped state groups that match at least one teacher
action, and `dapo/awm/skipped_all_oracle_group_rate`, the fraction of fully
skipped state groups whose every candidate matches the teacher multiset.

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

This validates 1,000×10 task cardinality, uniqueness, and pure-code verifier
coverage, then writes:

- `awm_all.parquet`: 1,000 environments / 10,000 tasks; and
- `manifest.json`: source hashes, revision, selection rule, counts, and the
  historical logical `all`/`dev`/`smoke` task IDs.

Only the complete Parquet is materialized. Development, smoke, and evaluation
slices use a deterministic verified-pool selection or an explicit task limit.

Dev and smoke selection uses only SHA-256 ranks of scenario/task IDs. It never
uses expert output, verifier outcome, or student performance.

## Build the deterministic training pool

Selection renders all 10,000 fixed native-tool prompts with the exact Qwen3
tokenizer. At the 16,000-token cutoff the pinned audit must reproduce all of
these values or fail:

- 9,380 eligible tasks;
- 938 environments with at least one eligible task; and
- 938 environments whose complete ten-task set is eligible.

All 9,380 context-eligible tasks enter the deterministic filter. Selection does
not call an expert, judge task quality, or run an end-to-end trajectory. Start
the pinned AWM server, then run:

```bash
bash examples/awm/data/run_selection.sh
```

The resumable output under `runs/awm_context_selection` contains the complete
10K prompt audit, `awm_context_candidates.parquet`, and a hash-bound manifest.
`cli/select_tasks.py --verify-only --output-dir ...` checks it without AWM.

Run the deterministic integrity filter:

```bash
bash examples/awm/data/run_integrity_audit.sh
```

The filter validates all 9,380 candidates against the pinned task, sample,
database-schema, and executable code-verifier sources. SQL-verifier records are
retained as diagnostics, but SQL-only defects cannot quarantine a task because
training and evaluation execute the code verifier. It also performs a native
reset, exact task check, raw/canonical tool-schema hash check, JSON Schema
validation, and untouched code verification with three infrastructure attempts.
Missing/conflicting active code records, compile errors, changed schemas/tasks,
and already-complete no-op states are deterministic quarantine reasons. Runtime
task/schema identity mismatches are retried on fresh resets before quarantine.
Timeout, server, and runtime verifier failures instead become
`infrastructure_pending`.

The main path fixes `SKIP_JUDGE=1`, so it makes no DeepSeek calls. Explicit
`SKIP_JUDGE=0` remains a diagnostic-only review mode and does not add an expert
success gate.

The audit still records `pass`, `needs_review`, and
`infrastructure_pending` for diagnosis, but these three statuses all enter
`awm_training_pool.parquet`. Only deterministic `quarantine` tasks enter
`rejected_prefilter_task_ids.json`. Expert success/failure is not a membership
criterion. The legacy `awm_integrity_filtered.parquet` remains a pass-only
diagnostic artifact. `--verify-only` validates every hash and ordered task ID.

The historical protocol-v4 artifact partitioned 6,617 `pass`, 809
`needs_review`, 2 `infrastructure_pending`, and 1,952 `quarantine`. Its active
pool contained 7,428 tasks from 936 environments. The two remaining
context-eligible environments had no task left after deterministic quarantine.

A follow-up stratified audit found that this protocol-v4 pool is
**superseded**, not a formal-data freeze. Exactly 399 tasks were rejected only
for conflicting SQL verifiers and one only for a missing SQL entrypoint,
although the executable training/evaluation protocol uses the code verifier;
all 400 have a unique code verifier, matching runtime task/schema, and an
incomplete no-op result. One additional schema-hash mismatch reproduced as a
match on three fresh resets. The remaining 20 invalid-canonical-schema findings
came from duplicate entries in JSON Schema `required` arrays.

Selection protocol v6 and integrity protocol v5 implement the corrected policy:
SQL-only findings are warnings, duplicate `required` entries are repaired, and
identity mismatches require three failed fresh resets. A targeted replay of the
20 repaired-schema tasks found 16 incomplete and four already complete under
the code verifier. Reclassifying the old evidence plus that replay projects
6,985 `pass`, 858 `needs_review`, 2 `infrastructure_pending`, and 1,535
`quarantine`, for a 7,845-task active pool. These counts are an audit projection;
the regenerated, hash-bound manifest is authoritative. Protocol-v4 selection
and pool artifacts are rejected by the current launcher and cannot be silently
reused.

## Optional one-pass expert environment screening

The deterministic pool may be screened once with the same native DeepSeek tool
calling used by evaluation. This is not an expert-success gate: both expert
successes and ordinary policy failures enter `awm_expert_screened_pool.parquet`.
Only a strong tool/verifier environment error reproduced after a fresh reset and
exact structured-action-prefix replay becomes `rejected_environment`. Transient
or ambiguous infrastructure errors remain `infrastructure_pending`. Screening
uses `history_window=6`, 20 decisions, and the training-aligned 32K budget split
of 27,904 prompt plus 4,096 response tokens; unattempted tasks remain `pending`.

The output is append-resumable and binds the complete deterministic candidate
pool. `MAX_NEW_TASK_FRACTION` limits only the current invocation, so a 5% cost
pilot can later resume in the same output directory with a different limit or
no limit:

```bash
MAX_NEW_TASK_FRACTION=0.05 RESUME=auto \
  bash examples/awm/screening/run_expert_screening.sh
```

`summary.json` and `screening_manifest.json` report request, prompt, DeepSeek
cache-hit/cache-miss, completion, and total-token usage. A partial screening
pool contains only already accepted tasks and is hash-verifiable by the training
launcher; the manifest's `pending` count makes partial coverage explicit.

During semantic training, tool/verify infrastructure errors are retried once by
resetting the same task with the same seed and replaying the exact structured
tool-call prefix without an LLM. If replay reproduces the failure or remains
ambiguous, only the current state group is masked and only that episode ends;
earlier healthy groups remain trainable, and the task is not blacklisted for
future episodes. Both cases are appended to `runtime_failures.jsonl` for
diagnosis. Ordinary model errors and unsuccessful outcomes remain training
data. Terminal outcome is logged only and is never added to semantic reward.

## Verifier-reliable semantic review

After one-pass expert screening reaches `pending=0`, build a hash-bound review
plan. It covers every policy failure and replay-confirmed environment failure,
plus a deterministic 10% success control stratified by prompt length and
decision count:

```bash
$PYTHON examples/awm/screening/semantic_audit.py plan \
  --data runs/awm_deterministic_filter/awm_training_pool.parquet \
  --candidate-manifest runs/awm_context_selection/candidate_manifest.json \
  --integrity-manifest runs/awm_deterministic_filter/integrity_manifest.json \
  --screening-dir runs/awm_expert_screening \
  --awm-data-dir /mnt/public2/yuanhuining/repos/openenv-awm-cache \
  --output-dir runs/awm_semantic_audit
```

Evidence capture is serial because upstream AWM scenario-port allocation has a
time-of-check/time-of-use window. Each packet fresh-resets with seed 300,
replays recorded structured actions, runs the code verifier, records the
initial/final SQLite diff, and safely removes only the validated retained
`/tmp/openenv_awm_<scenario>_*` session. Canonical/raw tool schemas, every tool
observation, and the verifier observation must reproduce their screened
signatures exactly; drift remains `pending` and never enters the review queue.
The 289 MB expert trial file is indexed and read one record at a time rather
than loaded into memory.

```bash
$PYTHON examples/awm/screening/semantic_audit.py capture \
  --output-dir runs/awm_semantic_audit \
  --awm-base-url http://127.0.0.1:8000
```

`next --slot A` and `next --slot B` emit independent Codex prompts. Submit each
JSON judgment through `record`. Exclusion requires matching A/B verdicts,
confidence at least 0.90, and a shared packet cohort key. Avoidable environment
bugs remain included. A defect found in a successful control expands review to
the cited environment/verifier/error cohort. The canonical contract is
committed in `screening/CODEX_REVIEW_PROMPT.md`.

```bash
$PYTHON examples/awm/screening/semantic_audit.py finalize \
  --output-dir runs/awm_semantic_audit \
  --all-data data/awm/awm_all.parquet \
  --all-manifest data/awm/manifest.json
```

Finalization partitions all 10,000 tasks into `included`, `excluded`, `pending`,
and `out_of_context`, and creates one `awm_verified_task_pool.parquet` shared by
semantic and outcome methods. The pool verifier re-derives every A/B consensus
from its bound evidence and judgment files and requires the exact prepared
10,000-task manifest used by context selection.

If the pinned system/task/tool schemas plus the newest complete action-result
exchange exceed `data.max_prompt_length`, the exchange is never truncated.
Teacher-first preflight instead terminates and masks only that state, closes its
environment session, and continues the other states in the batch. These events
are separate from runtime failures and are reported by
`env/context_overflow_rate`, `env/context_overflow_prompt_tokens_mean`, and
`env/context_overflow_excess_tokens_mean`, with per-state diagnostics in the
run log.

Training the deterministic pool is explicit:

```bash
TRAIN_DATA=runs/awm_deterministic_filter/awm_training_pool.parquet \
TRAIN_SELECTION_MANIFEST=runs/awm_deterministic_filter/integrity_manifest.json \
MODEL_PATH=/mnt/public2/yuanhuining/models/Qwen3-4B \
  bash examples/awm/train/run_semantic.sh
```

The launcher hash-verifies the pool, then materializes an exact-length
deterministic cyclic schedule. The default 200×64 schedule contains 12,800
rows. The regenerated active pool appears once before the deterministic prefix
repeats; the exact repeat count comes from its manifest. With `shuffle=false`
and a full-batch schedule, `drop_last` omits nothing.

The old 938/1,000-task selection, integrity, qualification, and expert-pilot
protocols are no longer supported by the main branch. If their compact metadata
is retained for provenance, keep it separately under `runs/legacy_manifests/`;
current selection, filtering, and training never depend on it.

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

For a non-smoke run, the launcher defaults to the verified deterministic pool
under `runs/awm_deterministic_filter`; it no longer uses `TRAIN_SPLIT=all`
implicitly. Formal defaults are 200 optimizer steps, 64 tasks per step, four
student candidates per state, two A800 GPUs, save every 10 steps, and validation
at step 0 and every 20 steps. Checkpoints are retained without a default cap.
Set `USE_RAW_SPLIT=1` only for an explicit diagnostic run. Every formal pool
launch verifies the source hashes, manifest, Parquet hash, and ordered task IDs.

`TRAIN_TASK_FRACTION` or `TRAIN_TASK_COUNT` selects a reproducible ordered
prefix after deterministic quarantine. Fractions are defined against the 9,380
context-eligible tasks, not against the active-pool size, so `0.1` selects 938
tasks when that many remain. Filtering preserves source order but does not
rebalance after removing quarantine rows. It is therefore a deterministic 10%
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

The isolated outcome baseline has no DeepSeek dependency:

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
Training and standalone AWM evaluation retain the six most recent action
exchanges by default. Set `HISTORY_WINDOW` to any non-negative integer to run a
different context-window ablation; `max_steps` remains independently fixed at
20 for the main training protocol.
Each new semantic run evaluates fixed-composition, complete Tau validation
batches at step 0 and every 20 steps with the resident training vLLM and the
same temperature, top-p, and top-k as training. With the default
`VAL_BATCH=16`, Airline-only evaluates 48 of 50 official `base` tasks. Set
`TAU_VAL_DOMAINS=airline,retail` for a fixed 5-Airline/11-Retail template that
evaluates 50 plus 110 tasks, or `VAL_BEFORE_TRAIN=false` only when intentionally
skipping the baseline. Validation uses seed 300, one trial per task, and no
teacher; its manifest records evaluated and dropped rows. Full official-split
evaluation is a separate manual native-runner step, never an automatic training
finalizer. On save/validation overlaps, the checkpoint is written first.
The two-GPU default uses `SP_SIZE=2`. Both AWM variants use the paper setting
`entropy_coeff=0`; they also disable the otherwise metric-only full-vocabulary
entropy recomputation, which is not part of the loss and is prohibitively large
at this context length. The actor dynamic-microbatch default is 16,384 tokens
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
the end:

```bash
SPLIT=all TASK_LIMIT=8 MODEL_PATH=/mnt/public2/yuanhuining/models/Qwen3-4B \
  bash examples/awm/eval/run_eval.sh
```

Set `START_VLLM=0 API_BASE=http://host:port/v1` to use an already persistent
server. `SEED` defaults to 300 and is part of every result and the strict run
identity. Evaluation incrementally writes raw per-task trajectories and verifier
results before producing a summary. `--resume` is accepted only when the
complete protocol identity matches.
For local checkpoints, that identity hashes the contents of every artifact file,
including all weight shards.

To inspect Qwen3-4B function-call parsing on the expert-qualified diagnostic
tasks without loading/offloading the training rollout engine, use the native
evaluation process:

```bash
DATA_FILE=runs/awm_expert_screening/awm_expert_screened_pool.parquet \
SELECTION_MANIFEST=runs/awm_expert_screening/screening_manifest.json \
MODEL_PATH=/mnt/public2/yuanhuining/models/Qwen3-4B \
TASK_LIMIT=32 SPLIT=all bash examples/awm/eval/run_eval.sh
```

Its summary reports verifier success, action-kind counts, parse failures,
schema-valid tool calls, tool execution errors, decision counts, and token use;
the JSONL retains every raw action and parsed action for manual inspection.
