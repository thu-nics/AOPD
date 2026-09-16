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
- Tool rewards share AWM/EnvScaler's deterministic equivalence followed by a
  source-aware matcher for every unresolved same-tool argument difference.
  The matcher sees public history, schema, both argument objects and the native
  tool/helper/type source, never task answers or DB snapshots.
  Verified native defaults may match omitted arguments; changed identifiers,
  quantities and native literal constraints are not accepted as paraphrases.
  Reviewed source-hash-bound rules treat retail return `item_ids` as a multiset
  and exchange `(item_ids, new_item_ids)` as a multiset of **pairs**: order may
  change, but multiplicity and old-to-new mapping are preserved.
  Execution preserves explicit nulls and omitted fields. In addition,
  `transfer_to_human_agents.summary` is ignored for reward matching: every valid
  teacher transfer vote matches a valid student transfer regardless of wording.
  Schema validation still applies. Original summaries, raw-action diversity,
  execution, history and repetition diagnostics are unchanged. Tool-matcher
  cache protocol 3 partitions verdicts by source/rules/schema/public context.
  Teacher cache identity is unchanged by this update; older-generation records
  still require explicit revalidation/import. This changes reward semantics, not teacher generation
  or native outcome evaluation, so continuing an old run is an intervention
  rather than an unchanged-protocol resume.
  See [shared action-equivalence protocol](../../docs/tool_action_equivalence.md)
  for the rule registry, failure handling and native counterfactual tests.
- Each schema-invalid teacher vote is retried independently up to two times;
  valid peer votes are never resampled. Only valid votes are cached. A partial
  exact-state cache is usable immediately and later refills only its missing
  vote indices. One or two valid votes remain trainable while reward scaling
  retains the fixed K=3 denominator. A state with no valid vote discards that
  current state group and ends only its owning trajectory; earlier valid groups
  and the rest of the batch remain trainable.
- Message matching checks exact equality, then the persistent evidence-scoped
  cache, then one API request per unique pair, in parallel (default concurrency
  32). Duplicate teacher votes are restored when summing rewards. The shared
  constraint-first prompt distinguishes questions/plans from completed actions.
  Message protocol **5** partitions verdicts by prompt, identity and decoding.
  DeepSeek uses thinking-low / 32,768 tokens for both messages and tool arguments,
  with no temperature/top-p overrides. The default local Qwen matcher retains
  thinking / temperature 0 / 8,192 tokens for messages and non-thinking / 1,024
  tokens for tool arguments. Teacher sampling/cache identity is unaffected.
  API/JSON/truncation failures are missing supervision, never false verdicts.
- Agentic OPD caps an unsupported fixed transfer notice at reward **0** until
  an actual transfer tool call returns success. It remains a valid trainable
  message, with raw semantic reward/votes retained; no execution veto or task
  quarantine is added. `TAU_TRANSFER_REWARD_GUARD=false` disables this rule.
  Native evaluation and Outcome-GRPO do not use the guard.
- Set env.tau.oracle.teacher_cache_import_paths=[/old/run/cache/teacher.jsonl]
  to import compatible votes into a new writable cache. Source files stay
  read-only; identity, prompt and current schema are checked at the exact state.
  Old context-free matcher decisions are never imported.
- A semantic-matcher infrastructure failure masks only that current state
  group, then ends its owning trajectory. Earlier valid groups in that
  trajectory remain trainable. The failure is never converted into a negative
  semantic judgment, and no student candidate is executed for the failed
  group.
- Outcome-GRPO uses four independent full trajectories per task and terminal
  trajectory reward. It never enters the teacher/state-group path.
- Student and teacher receive the same native Tau conversation and tool schemas.
  Privileged teacher context is opt-in and disabled by default.
- Prompt history keeps the newest complete exchanges that fit the token budget.
  An irreducibly oversized current state group is discarded and its owning
  trajectory is ended rather than silently left-truncated; earlier valid groups
  remain trainable.
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
| `TAU_NATIVE_LOG_LEVEL` | Tau native worker logging; defaults to `WARNING` to suppress full per-turn message dumps |
| `TAU_TEACHER_API_BASE` | OpenAI-compatible teacher endpoint; required by Agentic OPD |
| `TAU_TEACHER_API_KEY` | Teacher endpoint key; defaults to `EMPTY` |
| `ORACLE_CACHE` | Exact-state teacher cache; defaults to `<run>/cache/teacher.jsonl` |
| `ORACLE_MATCHER_CACHE` | Persistent semantic-pair cache; defaults to `<run>/cache/matcher.jsonl` |
| `TAU_MATCHER_ENABLE_THINKING` | Optional reasoning override; `null` selects provider default |
| `TAU_MATCHER_MAX_TOKENS` | Optional reasoning + answer budget; `null` selects provider default |
| `TAU_MATCHER_REASONING_EFFORT` | DeepSeek thinking effort; default `low` |
| `TAU_MATCHER_MAX_CONCURRENT_REQUESTS` | Matcher concurrency, default `32`, independent of teacher |
| `TAU_TRANSFER_REWARD_GUARD` | Training-only unsupported fixed transfer notice reward cap; default `true` |
| `TAU_MASK_MATCHER_REQUIRED_GROUPS` | Programmatic-only ablation; default `false`. Any unresolved pair masks the **whole state group**, bypassing matcher API and cache. |
| `TAU_MATCHER_PROVIDER` | `openai-compatible` (default) or `deepseek`; affects both message and tool-argument matching |
| `TAU_MATCHER_MODEL` / `TAU_MATCHER_API_BASE` | Optional independent matcher identity; inherit teacher values when unset |
| `TAU_MATCHER_API_KEY_ENV` | Matcher key variable name; inherits teacher only for the same endpoint, otherwise must be explicit |
| `TAU_TEACHER_VALIDITY_MAX_RETRIES` | Extra retries for each schema-invalid vote; defaults to `2` |

With `TAU_MASK_MATCHER_REQUIRED_GROUPS=true`, exact/canonical/source-proven
normalizations (including transfer-summary omission) remain available. A masked
group executes one uniformly sampled **valid student candidate** and continues;
its zero reward is an unscored placeholder, never a negative training label.
Other groups in that trajectory can still train normally. This applies to
message and same-tool argument matching, even when just one pair is unresolved.
Teacher K=3, frequency rewards and compatible teacher-cache imports are unchanged;
no matcher credential is needed. Monitor `episode/env/matcher_required_group_rate`
(also per domain) and `dapo/effective_state_groups`: fewer eligible groups may
reduce optimizer updates. This is a training-data/rollout ablation, not an
equivalent replacement for semantic matching. Start it as a separate fresh run.

For an OpenAI-compatible vLLM user endpoint, retain the `openai/` LiteLLM
prefix in `TAU_USER_MODEL`; the teacher client uses the raw served model ID.

The validated sampling protocol is still Qwen-oriented: the teacher uses
thinking with temperature 0.6, top-p 0.95, top-k 20 and 8,192 output tokens;
the Qwen3.5 user simulator uses thinking with temperature 1.0, top-p 0.95,
top-k 20, presence penalty 1.5 and 8,192 output tokens. Override these only as
an explicit protocol change.

For a DeepSeek thinking-low matcher while keeping the Qwen teacher unchanged,
export `DEEPSEEK_API_KEY` and prepend these variables to the training command:

```bash
TAU_MATCHER_PROVIDER=deepseek \
TAU_MATCHER_MODEL=deepseek-v4-flash \
TAU_MATCHER_API_BASE=https://api.deepseek.com \
TAU_MATCHER_API_KEY_ENV=DEEPSEEK_API_KEY \
TAU_MATCHER_ENABLE_THINKING=true \
TAU_MATCHER_REASONING_EFFORT=low \
TAU_MATCHER_MAX_TOKENS=32768 \
bash examples/tau_bench/train/run.sh
```

This uses DeepSeek's native `thinking={"type":"enabled"}` and low effort,
without temperature/top-p overrides. Keep the student, teacher and user variables
from the normal training command. Use a new run directory for a fresh
experiment and reuse only compatible teacher caches, not old matcher verdicts.

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

The shared Agentic OPD optimization defaults match AWM/EnvScaler: learning rate
`1e-6` with zero warmup, weight decay `0.01`, symmetric PPO clipping at `0.2`,
no overlong reward shaping, token-mean loss, and sampled-token entropy logging
without full-vocabulary entropy recomputation. These remain independently
overridable for controlled ablations.

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
