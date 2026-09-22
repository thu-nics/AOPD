# Tau Bench training and evaluation

This directory contains the canonical Tau Airline/Retail/Telecom entry points for
Agentic OPD, outcome-GRPO, and native evaluation. Machine-specific paths and
service endpoints are intentionally not stored here.

## Protocol

- Tau is pinned to commit `17e07b1da2bbc0cadfddeea36412686e0604127b`
  plus the checked-in optional-voice compatibility patch.
- Training supports the complete official `train` splits: Airline 30, Retail 74,
  Telecom 74. The legacy default remains Airline/Retail only.
  There is no qualification or expert-success filter.
- Default rollout batches contain 16 tasks (5 Airline, 11 Retail). Domain quotas
  are configurable. An RL step may
  contain several optimizer minibatches; a fully equal-reward batch skips updates.
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
- Agentic OPD penalizes an unsupported fixed transfer notice with reward **−1** until
  an actual transfer tool call returns success. It remains a valid trainable
  message, with raw semantic reward/votes retained; no execution veto or task
  quarantine is added. `TAU_TRANSFER_REWARD_GUARD=false` disables this rule.
  Native evaluation and Outcome-GRPO do not use the guard.
- Set env.tau.oracle.teacher_cache_import_paths=[/old/run/cache/teacher.jsonl]
  to import compatible votes into a new writable cache. Source files stay
  read-only; model identity, decoding, prompt and current schema are checked at
  the exact state. Teacher transport endpoints may differ when they serve the
  same model; the source endpoint is retained as provenance. Matcher cache
  endpoints remain part of identity and matcher profiles do not share verdicts.
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
| `TAU_TEACHER_SOURCE` | `external` (default) or `self`, using the current student rollout weights |
| `TAU_TEACHER_MODEL` | Raw model ID served by the teacher endpoint; required by external-teacher Agentic OPD |
| `TAU_NATIVE_LOG_LEVEL` | Tau native worker logging; defaults to `WARNING` to suppress full per-turn message dumps |
| `TAU_TEACHER_API_BASE` | OpenAI-compatible teacher endpoint; required only for external-teacher Agentic OPD |
| `TAU_TEACHER_API_KEY` | Teacher endpoint key; defaults to `EMPTY` |
| `ORACLE_CACHE` | Exact-state teacher cache; defaults to `<run>/cache/teacher.jsonl` |
| `ORACLE_MATCHER_CACHE` | Persistent semantic-pair cache; defaults to `<run>/cache/matcher.jsonl` |
| `TAU_MATCHER_ENABLE_THINKING` | Optional reasoning override; `null` selects provider default |
| `TAU_MATCHER_MAX_TOKENS` | Optional reasoning + answer budget; `null` selects provider default |
| `TAU_MATCHER_REASONING_EFFORT` | DeepSeek thinking effort; default `low` |
| `TAU_MATCHER_MAX_CONCURRENT_REQUESTS` | Matcher concurrency, default `32`, independent of teacher |
| `TAU_TRANSFER_REWARD_GUARD` | Training-only unsupported fixed transfer notice reward `-1`; default `true` |
| `TAU_MASK_MATCHER_REQUIRED_GROUPS` | Programmatic-only ablation; default `false`. Any unresolved pair masks the **whole state group**, bypassing matcher API and cache. |
| `TAU_MATCHER_PROVIDER` | `openai-compatible` (default) or `deepseek`; affects both message and tool-argument matching |
| `TAU_MATCHER_MODEL` / `TAU_MATCHER_API_BASE` | Optional independent matcher identity; inherit teacher values when unset |
| `TAU_MATCHER_API_KEY_ENV` | Matcher key variable name; inherits teacher only for the same endpoint, otherwise must be explicit |
| `TAU_TEACHER_VALIDITY_MAX_RETRIES` | Extra retries for each schema-invalid vote; defaults to `2` |
| `AIRLINE_TRAJ` / `RETAIL_TRAJ` / `TELECOM_TRAJ` | Fixed worker-domain quotas; defaults `5/11/0` |
| `TAU_ABLATION` | `full` (default), `a1` validity-only, `a4` binary K3, `a5` random commit |
| `TAU_MATCHER_PROFILE` | `default` or frozen `qwen38_concise`; the latter requires non-thinking and max tokens 32,768 |
| `TAU_USER_REASONING_ENABLED` / `TAU_USER_TEMPERATURE` / `TAU_USER_TOP_P` | Explicit training-user protocol; defaults remain `true/1.0/0.95` |
| `TAU_INTERNAL_MAX_STEPS` | Native orchestration budget (including user/tool steps), distinct from agent decisions; default derives from the agent limit |
| `TAU_TEACHER_CACHE_IMPORT_PATHS` | JSON list of read-only source teacher caches |

### Self-AOPD (experimental)

Set `TAU_TEACHER_SOURCE=self` (synchronous vLLM rollout) and configure an explicit, fixed
`TAU_MATCHER_MODEL`, `TAU_MATCHER_API_BASE`, and `TAU_MATCHER_API_KEY_ENV`.
No teacher API/server is used. `TAU_USE_PRIVILEGED_TEACHER_CONTEXT=false/true`
selects the public-only S1 / privileged S2 experiment. Both use the same teacher
instruction, N=4 independent student candidates and K=3 independent teacher votes.
Self-teacher decoding must match stochastic student decoding, including the actual
rollout response budget; default teacher output is
4,096 tokens, unlike the external teacher's 8,192-token default.

Public training prompts retain their 24,576-token budget. An additional
`TAU_SELF_EXTRA_PROMPT_TOKENS=4096` reserves teacher-only instruction/task space
inside the existing 32,768-token rollout window. Teacher receives the **same
retained public conversation history**, with no separate history truncation.
Both teacher variants receive a natural-language rendering of the pinned Telecom
technical manual: the troubleshooting knowledge is retained, but customer-device
API names and signatures are removed (including customer-side payment references
in the main policy). Only the teacher's system-policy block is
changed; student/native eval prompts, agent tools, and actual history/tool results
are untouched. S1 is public-information-only, not a byte-identical student prompt.
S2 requires `TAU_SELF_CUSTOMER_BRIEFS=<reviewed frozen JSON>`. Offline briefs
contain third-person goals, constraints, conditional preferences and customer
knowledge availability, with source excerpts and per-record approval. No online
summarization is performed. Source drift or unreviewed briefs fail explicitly.
`TAU_SELF_PRIVILEGE_MODE=answer_conditioned` (default for privileged self teacher)
adds official train reference actions, target outcomes and communication requirements.
Full reference parameters are visible to teacher, but do not constitute identity
verification, consent or execution evidence. Customer operations and assertions
are deterministically converted to plain language; user-side API names, grading
metadata, full DBs and simulator instructions are not exposed. Unknown mappings
fail during the all-task startup audit. Missing answers are explicitly marked
unavailable, not invented or interpreted as an instruction to do nothing (Retail
train task 57 has this case). This uses additional answer supervision and must be
reported as answer-conditioned Self-AOPD, not customer-information-only training.
No online answer rewriting or live DB snapshot is needed.

The earlier `customer` and `customer_and_state` modes remain explicit diagnostic
alternatives; only the latter reads allowlisted customer-scoped live DB facts.
Both S1/S2 use identical public guidance and preserve required public consent.
In those earlier modes, private lookup identifiers use stable aliases unless grounded in retained user
messages or tool observations; assistant guesses do not count. Knowledge
availability remains visible (e.g. the customer can provide an email), even when
the actual value is hidden. This is not a new action-validity or reward rule.
Brief digest, projection, mode, train-answer digest and snapshot revision are part of protocol identity.
Neither student loss nor matcher inputs receive privileged notes. Startup checks
every official train task's overhead; runtime overflow discards only the current
group and ends its rollout. This prompt/projection revision requires a **new run
directory**; old Self-AOPD runs cannot resume under the changed protocol.
The paired direct-policy diagnostic compares public vs answer-conditioned Qwen3-4B
on 178 official train tasks, three trials per arm (1,068 trajectories), starting
with a 24-task/48-trajectory pilot. It is not a leaderboard evaluation; final native
evaluation never receives answers. The diagnostic keeps its existing remote
Qwen3.5 user and 20 agent decisions / 200 internal transitions.

Student and teacher rows share one inference call; teacher rows and extra prompt
padding are removed by request ID before training. Missing votes receive up to two retries;
partial sets retain denominator K=3. Votes are memoized **only within one RL
step** and never imported from disk or another run. `<run>/cache/self_teacher.jsonl`
is an audit log, not a reusable teacher cache. Revisions, context overhead,
clipping and retry counts are recorded. Existing matcher caching remains separate.
The programmatic-only masking ablation can also disable the fixed matcher;
then neither teacher nor matcher API credentials are required (the user simulator
still needs its endpoint). Native evaluation is unchanged. Run GPU smoke and a real-state teacher quality
audit before treating this experimental path as ready for a full run.

With `TAU_MASK_MATCHER_REQUIRED_GROUPS=true`, exact/canonical/source-proven
normalizations (including transfer-summary omission) remain available. A masked
group executes one uniformly sampled **valid student candidate** and continues;
its placeholder rewards produce no loss, never an unmatched training label.
Other groups in that trajectory can still train normally. This applies to
message and same-tool argument matching, even when just one pair is unresolved.
Teacher K=3, frequency rewards and compatible teacher-cache imports are unchanged;
no matcher credential is needed. Monitor `episode/env/matcher_required_group_rate`
(also per domain) and `dapo/effective_state_groups`: fewer eligible groups may
reduce optimizer updates. This is a training-data/rollout ablation, not an
equivalent replacement for semantic matching. Start it as a separate fresh run.

The transfer guard requires a real `transfer_to_human_agents` call with its linked
successful tool result before the fixed handoff notice. Case/whitespace and
whole-message `<response>`/bold wrappers are accepted; quotations, offers and
questions are not violations. The penalty overrides matching **before** commit,
but preserves native action validity, raw teacher verdicts and execution. An
all-`-1` group still has no relative signal and is skipped normally. Monitor
`episode/env/transfer_without_tool_candidate_rate` and
`dapo/transfer_without_tool_positive_advantage_rate` (the latter is emitted only
when violations actually participate in training; expected value: zero).
Earlier code capped this reward at zero and did not forward its candidate metric
correctly. This is a reward-protocol change; old metric zeros are not evidence of
zero violations. Teacher/matcher caches remain reusable because their inputs and
equivalence semantics are unchanged. Native eval and outcome GRPO are unaffected.

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
