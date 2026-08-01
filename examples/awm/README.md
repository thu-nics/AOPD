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
- Training and internal evaluation use at most 20 student decisions and the
  pure-code verifier. Expert qualification uses AWM SQL verification augmented
  by the same DeepSeek model as judge; judge timeouts, server failures, and
  unavailable verifiers are retried as infrastructure errors instead of policy
  failures.

The direct DeepSeek API model ID is `deepseek-v4-flash`. Teacher calls enable
thinking with `reasoning_effort=max`. DeepSeek ignores `temperature` and `top_p`
in thinking mode, so the teacher protocol does not send or cache those ineffective
controls. Matcher calls disable thinking and use temperature zero. Training metrics
separate live API requests/tokens from cache hits and loaded-cache inventory, so a
resumed run does not report historical cache cost as new usage. The API key is read
from `DEEPSEEK_API_KEY` and is never written to a dataset, manifest, or cache.
Each cached response records the provider-returned model and system fingerprint;
an identity change within one cache/run fails instead of mixing teacher versions.

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

## Select and qualify the training subset

The expert-screening pool is a deterministic, environment-balanced 1,000-task
subset of the full public data. Selection first renders every task's fixed
native-tool prompt with the exact Qwen3 tokenizer. At the 16,000-token cutoff the pinned
dataset audit must reproduce all of these values or fail:

- 9,380 eligible tasks;
- 938 environments with at least one eligible task; and
- 938 environments whose complete ten-task set is eligible.

Each selected task also passes a native reset, fresh tool-schema check, and
no-op pure-code verifier preflight. Selection gives each viable environment one
task before assigning a second task to the small remainder, and never assigns
more than two. Start the AWM server, then run:

```bash
bash examples/awm/scripts/run_selection.sh
```

The output under `runs/awm_selection_native_canonical_1k` is resumable and contains the complete
10K native-prompt audit, preflight records, the 1K Parquet, and a hash-bound candidate
manifest. `cli/select_tasks.py --verify-only --output-dir ...` checks the artifacts
without contacting AWM.

Before qualification, run the independent integrity filter from the
`deepseek_api` tmux shell:

```bash
bash examples/awm/scripts/run_integrity_audit.sh
```

The filter validates all 1,000 candidates against the pinned task, sample,
database-schema, pure-code-verifier, and SQL/code-augmented-verifier sources.
It also performs a native reset, exact task check, raw/canonical tool-schema
hash check, JSON Schema validation, and untouched pure-code verification with
three infrastructure attempts. Missing/conflicting source records, compile
errors, changed schemas/tasks, and already-complete no-op states are deterministic
quarantine reasons. Timeout, server, and runtime verifier failures instead become
`infrastructure_pending`.

At most 64 tasks enter semantic calibration: the four observed pilot cases when
present, statically suspicious tasks, and 16 clean controls distributed across
prompt-length quartiles. DeepSeek reviews each independently twice with thinking
and `reasoning_effort=max`; automatic quarantine requires two `infeasible`
verdicts at confidence >=0.9 with the same defect kind affecting the SQL
protocol. Judge prompts are hard-capped at 24K Qwen tokens and the max-thinking
response budget defaults to 16K tokens; empty/truncated/invalid JSON responses
are retried and recorded only as infrastructure diagnostics, without retaining
reasoning text. Disagreement or uncertainty becomes `needs_review`. The filtered pool
contains only `pass` tasks, does not backfill toward 1,000, and is hash-bound to
the selection manifest. `--verify-only` validates every output hash and ordered
task ID without contacting AWM or DeepSeek.

Qualification runs the DeepSeek expert independently with seeds 300--303. A
task from `runs/awm_integrity_native_canonical_1k` is retained only after 4/4
successful SQL+DeepSeek-judge outcomes. A policy
failure stops that task early; infrastructure failures receive up to three
attempts and remain separately classified rather than being counted as policy
failures. Calls for one task are sequential while tasks run concurrently. Raw
reasoning, actions, tool results, verifier output, returned provider identity,
and token usage are persisted after every trial, so a stopped run resumes
without repeating completed calls:

```bash
# Run from the deepseek_api tmux shell so DEEPSEEK_API_KEY is inherited.
bash examples/awm/scripts/run_qualification.sh
```

For a priced pilot, set `MAX_NEW_TASKS=8`; rerunning later with the same output
directory and no limit continues the remaining candidates. The final outputs
include every 4/4-qualified task, an environment-balanced batch-size-8 training
file, and a Qwen diagnostic manifest of up to 32 distinct environments across
four native-prompt-length quartiles. Training the filtered set is explicit:

```bash
TRAIN_DATA=runs/awm_expert_qualification_native/awm_expert_qualified_train_b8.parquet \
TRAIN_SELECTION_MANIFEST=runs/awm_expert_qualification_native/qualification_manifest.json \
MODEL_PATH=/mnt/public2/yuanhuining/models/Qwen3-4B \
  bash examples/awm/scripts/run_semantic.sh
```

The launcher verifies the qualified Parquet hash and ordered task IDs before
training. It derives epoch length from that file, not from the original 10K
split.

## Start AWM

AWM scenario code is trusted research code and is not isolated inside the server
process beyond the outer machine/container boundary. Run it only on a suitable
development host.

```bash
export AWM_SOURCE_DIR=/mnt/public2/yuanhuining/repos/openenv-awm
export AWM_CACHE_DIR="/mnt/public2/yuanhuining/repos/openenv-awm-cache"
bash examples/awm/scripts/start_server.sh
```

The launchers expect `http://127.0.0.1:8000/stats` to be healthy.

## Train

Run semantic training from the `deepseek_api` tmux shell so the pane-local
`DEEPSEEK_API_KEY` is inherited:

```bash
MODEL_PATH=/mnt/public2/yuanhuining/models/Qwen3-4B \
  bash examples/awm/scripts/run_semantic.sh
```

The normal launcher defaults to `TRAIN_SPLIT=all` and derives 1,250 optimizer
steps at batch size 8, so its deterministic sequential sampler covers all 10,000
tasks exactly once. `TRAIN_STEPS` remains an explicit override. Set
`TRAIN_SPLIT=dev` only for integration work or short development runs. Every launch
strictly verifies the fixed source hashes, manifest, and ordered Parquet task IDs.

One-step development smoke:

```bash
SMOKE=1 MODEL_PATH=/mnt/public2/yuanhuining/models/Qwen3-4B \
  bash examples/awm/scripts/run_semantic.sh
```

The isolated outcome baseline has no DeepSeek dependency:

```bash
MODEL_PATH=/mnt/public2/yuanhuining/models/Qwen3-4B \
  bash examples/awm/scripts/run_outcome.sh
```

All machine-specific paths, batch sizes, GPU settings, run directories, and
server URLs are environment-variable overrides in `run_training.sh`. The two-GPU
default uses `SP_SIZE=2`. Both AWM variants use the paper setting
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
DATA_FILE=runs/awm_expert_qualification_native/awm_expert_qualified_all.parquet \
SELECTION_MANIFEST=runs/awm_expert_qualification_native/qwen_diagnostic_manifest.json \
MODEL_PATH=/mnt/public2/yuanhuining/models/Qwen3-4B \
  bash examples/awm/scripts/run_eval.sh
```

Its summary reports verifier success, action-kind counts, parse failures,
schema-valid tool calls, tool execution errors, decision counts, and token use;
the JSONL retains every raw action and parsed action for manual inspection.
