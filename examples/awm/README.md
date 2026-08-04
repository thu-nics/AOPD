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
- Student context: 32,000 tokens total, split into 29,952 prompt tokens and
  2,048 response tokens.
- The system message and task are pinned. At most three complete recent
  action/result exchanges are retained. The exact budget-trimmed chat and the
  same actual environment-tool schemas are shared by all four candidates and
  the teacher.
- There is no model-visible `list_tools` scaffold or nested `call_tool` meta
  tool. The adapter fetches schemas internally after reset, then Qwen3 and
  DeepSeek each receive those actual tools through their native function-calling
  interface. The adapter deterministically removes AWM's contradictory sibling
  `type: T` when the same node already declares `anyOf: [T, null]`; raw and
  canonical schema hashes plus every repair remain audit-visible. Because AWM's
  server still validates calls against the contradictory raw schema, explicit
  `null` on non-required nullable fields is canonicalized to argument omission
  before semantic matching and execution (equivalent to the upstream Python
  `Optional[T] = None` default).
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
- Training and internal evaluation retain the same at-most-three-exchange
  history and 20-decision action budget. The deterministic training pool uses
  the same native prompt and 16K fixed-scaffold cutoff but no expert-success
  qualification gate.
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
bash examples/awm/scripts/install_awm.sh
```

Local development defaults are centralized in `scripts/paths.sh`: `VENV_PATH`,
`AWM_SOURCE_DIR`, and `AWM_CACHE_DIR` point to the shared installation under
`/opt/venvs` and `/mnt/public2/yuanhuining/repos`. Existing `PYTHON`,
`OPENENV_ROOT`, and `AWM_DATA_DIR` overrides remain supported. Reusable AWM
logic lives in `agent_system.environments.env_package.awm`; Python files under
`examples/awm/cli` are compatibility CLI wrappers.

The installer refuses to mutate an existing OpenEnv checkout at another commit.
Network commands honor the standard proxy variables from the shell.

Prepare the complete pinned dataset and deterministic ID-only subsets:

```bash
$PYTHON examples/awm/cli/prepare_data.py \
  --data-dir "/mnt/public2/yuanhuining/repos/openenv-awm-cache" \
  --output-dir data/awm
```

This validates 1,000×10 task cardinality, uniqueness, and pure-code verifier
coverage, then writes:

- `awm_all.parquet`: 1,000 environments / 10,000 tasks;
- `awm_dev.parquet`: 32 environments / 256 tasks;
- `awm_smoke.parquet`: 4 environments / 8 tasks; and
- `manifest.json`: source hashes, revision, selection rule, counts, and all
  split task IDs.

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
bash examples/awm/scripts/run_selection.sh
```

The resumable output under `runs/awm_context_selection` contains the complete
10K prompt audit, `awm_context_candidates.parquet`, and a hash-bound manifest.
`cli/select_tasks.py --verify-only --output-dir ...` checks it without AWM.

Run the deterministic integrity filter:

```bash
bash examples/awm/scripts/run_integrity_audit.sh
```

The filter validates all 9,380 candidates against the pinned task, sample,
database-schema, pure-code-verifier, and SQL/code-augmented-verifier sources.
It also performs a native reset, exact task check, raw/canonical tool-schema
hash check, JSON Schema validation, and untouched pure-code verification with
three infrastructure attempts. Missing/conflicting source records, compile
errors, changed schemas/tasks, and already-complete no-op states are deterministic
quarantine reasons. Timeout, server, and runtime verifier failures instead become
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

During semantic training, tool/verify infrastructure errors are retried by
resetting the same task with the same seed and replaying the exact structured
tool-call prefix without an LLM. The same strong deterministic signature on
both executions writes `runtime_quarantine.jsonl`, masks every row from that
reset, and prevents future teacher/student actions for the task in that run.
Transient or ambiguous errors are `runtime_infrastructure_pending`: they mask
the affected trajectory but are not persisted as task defects. Ordinary model
errors and unsuccessful outcomes remain training data. Terminal outcome is
logged only and is never added to semantic reward.

Training the deterministic pool is explicit:

```bash
TRAIN_DATA=runs/awm_deterministic_filter/awm_training_pool.parquet \
TRAIN_SELECTION_MANIFEST=runs/awm_deterministic_filter/integrity_manifest.json \
MODEL_PATH=/mnt/public2/yuanhuining/models/Qwen3-4B \
  bash examples/awm/scripts/run_semantic.sh
```

The launcher hash-verifies the pool, then materializes an exact-length
deterministic cyclic schedule. The formal 200×64 schedule contains 12,800 rows:
all 9,380 verified tasks appear once before the first 3,420 tasks repeat. With
`shuffle=false` and a full-batch schedule, `drop_last` omits nothing.

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
bash examples/awm/scripts/start_server.sh
```

Context selection and deterministic integrity preprocessing expect this
standalone server at `http://127.0.0.1:8000`. Training does not reuse it.

## Train

Semantic training requires `DEEPSEEK_API_KEY` for the teacher. The Tau
user-simulator provider is selected by `TAU_USER_LLM`; its default OpenRouter
model requires `OPENROUTER_API_KEY`. To route every external request through
the official DeepSeek API, set
`TAU_USER_LLM=deepseek/deepseek-v4-flash`:

```bash
MODEL_PATH=/mnt/public2/yuanhuining/models/Qwen3-4B \
  bash examples/awm/scripts/run_semantic.sh
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
prefix after deterministic quarantine. The full 9,380 context-eligible pool is
stored in environment-balanced round-robin order, so a 10% experiment starts
with one task from each of the 938 eligible environments (minus any quarantined
tasks, filled by the next ordered tasks):

```bash
TRAIN_TASK_FRACTION=0.1 MODEL_PATH=/mnt/public2/yuanhuining/models/Qwen3-4B \
  bash examples/awm/scripts/run_semantic.sh
```

The run stores and hash-verifies its slice Parquet and manifest under
`runs/<UTC timestamp>/data/`. The two controls are mutually exclusive.

One-step development smoke, including two official Airline validation tasks:

```bash
MODEL_PATH=/mnt/public2/yuanhuining/models/Qwen3-4B \
  bash examples/awm/scripts/run_semantic_smoke.sh
```

The isolated outcome baseline has no DeepSeek dependency:

```bash
MODEL_PATH=/mnt/public2/yuanhuining/models/Qwen3-4B \
  bash examples/awm/scripts/run_outcome.sh
```

All machine-specific paths, batch sizes, GPU settings, run directories, and
managed-server host/port settings are environment-variable overrides in
`run_training.sh`. By default,
each launch writes under `runs/<UTC timestamp>/`; TensorBoard event files live in
that run's `tensorboard/` subdirectory instead of a repository-level
`tensorboard_log/`. `RUN_DIR` and `TENSORBOARD_DIR` remain explicit overrides.
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

`cli/eval_awm.py` uses `AWMEnv` and its native `reset`, `step`, `verify`, and
`done` methods directly. It fetches schemas internally once after reset and
passes the actual tools through Qwen3's native tool template; `list_tools` and
`call_tool` are not model actions. The vLLM launcher uses the `hermes` parser
matching this checkpoint's JSON-in-`<tool_call>` template, the `qwen3`
reasoning parser, and Qwen3's recommended thinking-mode sampling
(`temperature=0.6`, `top_p=0.95`, `top_k=20`) with the recorded evaluation
seed. It never creates the training Ray rollout stack. `scripts/run_eval.sh`
starts one OpenAI-compatible vLLM server, keeps the model
resident while every selected task runs, and stops only that server process at
the end:

```bash
SPLIT=smoke MODEL_PATH=/mnt/public2/yuanhuining/models/Qwen3-4B \
  bash examples/awm/scripts/run_eval.sh
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
DATA_FILE=runs/legacy/awm_expert_qualification_native/awm_expert_qualified_all.parquet \
SELECTION_MANIFEST=runs/legacy/awm_expert_qualification_native/qwen_diagnostic_manifest.json \
MODEL_PATH=/mnt/public2/yuanhuining/models/Qwen3-4B \
  bash examples/awm/scripts/run_eval.sh
```

Its summary reports verifier success, action-kind counts, parse failures,
schema-valid tool calls, tool execution errors, decision counts, and token use;
the JSONL retains every raw action and parsed action for manual inspection.
