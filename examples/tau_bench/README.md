# Tau Bench VPR

This directory contains Tau Airline/Retail training plus in-process and native
evaluation for this research fork.

## Protocol

- Source: `/mnt/public2/yuanhuining/repos/tau2-bench`, pinned to commit
  `17e07b1da2bbc0cadfddeea36412686e0604127b` plus the checked-in optional-voice
  compatibility patch.
- Training tasks: the complete official `train` split, Airline 30 and Retail
  74. There is no expert-success qualification gate.
- Periodic validation draws from the official `base` split with one trial per
  task, but intentionally materializes only complete, fixed-composition batches.
  With `VAL_BATCH=16`, Airline-only uses 16 Airline slots and evaluates 48 of
  50 tasks; Airline plus Retail uses 5/11 slots and evaluates 50 plus 110 tasks.
  The manifest records the fixed quota and every dropped tail row.
- Student prompt: Qwen's native ChatML function-calling format and actual Tau
  tool schemas.
- User simulator: `openrouter/qwen/qwen3.6-27b`, temperature 1, reasoning
  disabled.
- Expert: `deepseek/deepseek-v4-flash`, three independent requests per exact
  state. Concurrent requests for the same state use single-flight; a completed
  action set is reused from the run-local cache for the remainder of that run.
- Tau VPR preserves its existing set semantics after sampling: duplicate expert
  actions are deduplicated before reward matching. AWM separately preserves its
  K=3 multiset and frequency-weighted rewards.
- If the expert emits parallel tool calls, Tau executes only the first. A
  student multi-call output remains invalid under the single-action protocol.
- Training/evaluation caps are 20/30 agent decisions.
- Terminal reward is Tau's deterministic DB component multiplied by
  COMMUNICATE when applicable; LLM-judged NL assertions are excluded.

## Setup

Tau requires Python 3.12 or newer.

```bash
PYTHON=/opt/venvs/verl-agent/bin/python \
bash examples/tau_bench/install_tau2.sh
export TAU2_DATA_DIR=/mnt/public2/yuanhuining/repos/tau2-bench/data
export OPENROUTER_API_KEY=<OPENROUTER_API_KEY>
```

The installer keeps both source and Tau's dataset cache under the shared
`tau2-bench` checkout and installs it editable into the selected environment.

## Training

```bash
PYTHON=/opt/venvs/verl-agent/bin/python \
MODEL_PATH=<LOCAL_QWEN_MODEL> \
bash examples/tau_bench/run_tau_vpr.sh

PYTHON=/opt/venvs/verl-agent/bin/python \
MODEL_PATH=<LOCAL_QWEN_MODEL> \
bash examples/tau_bench/run_tau_outcome.sh
```

Both launchers materialize a deterministic cyclic schedule from the official
train tasks. Generated data, TensorBoard logs, caches, and checkpoints live
under `runs/<UTC timestamp>/`. The expert cache defaults to
`$RUN_DIR/cache/teacher.jsonl`; pass `ORACLE_CACHE` only when deliberate
cross-run reuse is desired. Set `SMOKE=1` for a one-step, two-decision smoke.

VPR uses four student candidates per visited state, commits exactly one
uniformly among the highest-reward candidates, and masks equal-reward groups.
Outcome uses four complete rollouts per task and trajectory-level GRPO.

## AWM periodic validation

The formal AWM semantic launcher calls the same in-process Tau adapter at step
0 and every 20 optimizer steps, using the training vLLM instance and sampling
parameters. Every worker keeps one domain for its lifetime, and every validation
batch uses the fixed domain quota recorded in the data manifest. A tail that
cannot fill that exact template is omitted.

```bash
# Default: 48 of 50 Airline base tasks (three complete 16-task batches)
bash examples/awm/train/run_semantic.sh

# Airline 50 + Retail 110 (ten complete 5+11 batches)
TAU_VAL_DOMAINS=airline,retail \
bash examples/awm/train/run_semantic.sh
```

On steps divisible by both save and validation frequency, the checkpoint is
written before validation. Formal AWM defaults are 200 steps, 64 tasks per
step, four candidate actions per state, save every 10, validate every 20, and
retain all checkpoints. The standalone smoke entry point is:

```bash
bash examples/awm/train/run_semantic_smoke.sh
```

## Evaluation

The lightweight in-process evaluator defaults to fixed 5-Airline/11-Retail
complete batches from the official `base` pools and is configurable through
`VALIDATION_DOMAINS`. It is intended for periodic or diagnostic comparison, not
final complete-split reporting:

```bash
PYTHON=/opt/venvs/verl-agent/bin/python \
MODEL_SPECS_FILE=<MODEL_REGISTRY_TSV> \
bash examples/tau_bench/run_tau_eval.sh
```

Training does not launch an automatic full evaluation at its final step. For
full-split final reporting, manually run the separate native Tau runner. It serves the student
with local vLLM and delegates task execution and deterministic scoring to Tau:

```bash
export DEEPSEEK_API_KEY=<DEEPSEEK_API_KEY>
MODEL_SPECS_FILE=<MODEL_REGISTRY_TSV> \
USER_MODEL=deepseek/deepseek-v4-flash \
DOMAINS="airline retail telecom-workflow" \
RUN_DIR=runs/tau_native_eval_final \
bash examples/tau_bench/run_tau_native_eval.sh
```

`AGENT_PROTOCOL=strict_native` is the final protocol. Use
`AGENT_PROTOCOL=training_compatible` only as a diagnostic comparison with the
training parser, in a separate run directory. Native results are checkpointed
in task shards and resume completed trials. `NUM_TASKS=1 DOMAINS=airline` is
the smallest native smoke. Supported native domains are `airline`, `retail`,
`telecom`, and Tau2's workflow-policy variant `telecom-workflow`. DeepSeek user
models use the official `DEEPSEEK_API_KEY` and provider-native
`thinking.type=disabled`; OpenRouter models retain their existing key and args.

## Metrics

- `episode/env/protocol_reward` and `episode/env/success_rate`: deterministic
  terminal task result.
- `episode/env/valid_action_rate`: schema-valid tool call or non-empty user
  message.
- `episode/env/oracle_hit_rate`: process-action match rate for VPR.
- `episode/env/oracle_cache_*`: cache lookups, hits, misses, single-flight
  waits, generated sets, and hit rate.
- `dapo/skipped_oracle_rate` and per-domain variants: oracle-candidate share
  among rows in fully skipped equal-reward groups.
