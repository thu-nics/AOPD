# Tau Bench VPR

This directory contains the reproducible Airline/Retail training pipeline used for the Tau Bench scalability experiment.

## Protocol

- Tau source: commit `17e07b1da2bbc0cadfddeea36412686e0604127b` plus the checked-in optional-voice compatibility patch.
- Domains: Airline and Retail.
- Student prompt: Qwen ChatML with native tool schemas.
- Student sampling: temperature `0.6`, top-p `0.95`, top-k `20`, min-p `0`.
- User simulator: `openrouter/qwen/qwen3.6-27b`, temperature `0`, reasoning disabled.
- Oracle policy: `deepseek/deepseek-v4-flash`, three independent seeded requests per state, `xhigh` reasoning, no temperature or top-p. If a provider ignores `parallel_tool_calls=false`, only the first tool call from that independent sample is retained.
- VPR reward: `+1` for an oracle-equivalent action, `0` for another valid action, and `-1` for an invalid action. One action is committed uniformly from the maximum-reward candidates.
- VPR batch: four Airline plus four Retail committed trajectories, with four student candidates at every visited state.
- Outcome batch: four Airline plus four Retail task groups, with four complete episodes per group. The deterministic terminal score is summed per episode, normalized across the four rollouts, and assigned to every generated turn in that episode.
- Terminal score: Tau's deterministic DB component, multiplied by COMMUNICATE when that component is in the task reward basis. Experimental LLM-judged NL assertions are excluded.
- Training/evaluation decision caps: `20`/`30` agent decisions.

Qualification evaluates all 30 Airline and 74 Retail training tasks with four trials each. A task is admitted only when all four trials succeed and no successful trial contains an illegal action. Training hard-fails unless at least 20 Airline and 50 Retail tasks qualify.

## Training Metrics

- `episode/env/protocol_reward` and `episode/env/success_rate` report the deterministic terminal task score described above.
- `episode/env/valid_action_rate` reports schema-valid tool calls or non-empty user messages.
- `episode/env/oracle_hit_rate` reports the fraction of VPR committed actions that match the sampled oracle set; it is zero for outcome training.
- In VPR, `episode/reward` is the accumulated process reward of committed actions and is not a terminal task-success metric.
- Equal-reward VPR state groups are masked from the policy loss. Outcome training uses standard trajectory-level GRPO: every sampled group is retained, equal terminal-score groups receive zero policy advantage, and no replacement sampling is performed.

## Setup

Tau requires Python 3.12 or newer.

```bash
PYTHON=<PYTHON_3_12> bash examples/tau_bench/install_tau2.sh
export OPENROUTER_API_KEY=<OPENROUTER_API_KEY>
```

The installer checks out the pinned source under `.cache/` by default and prints the required `TAU2_DATA_DIR`. The compatibility patch only removes eager imports of optional voice dependencies; it does not change Airline/Retail task or scoring logic.

## Qualification

```bash
PYTHON=<PYTHON_3_12> \
OUTPUT_DIR=data/tau_bench/qualification \
bash examples/tau_bench/run_qualification.sh
```

Qualification is resumable only with an identical protocol. It writes a protocol
fingerprint, trial records, successful deterministic-score trajectories, the
versioned oracle state cache, and `qualification_manifest.json` under `OUTPUT_DIR`.
Changing the expert, user simulator, seed, decision cap, or sampling protocol
requires a new `OUTPUT_DIR`; training hard-fails on a mismatched manifest.

## Training

```bash
PYTHON=<TRAINING_PYTHON> \
MODEL_PATH=<QWEN3_8B_MODEL_PATH> \
QUALIFICATION_MANIFEST=<QUALIFICATION_MANIFEST> \
bash examples/tau_bench/run_tau_vpr.sh

PYTHON=<TRAINING_PYTHON> \
MODEL_PATH=<QWEN3_8B_MODEL_PATH> \
QUALIFICATION_MANIFEST=<QUALIFICATION_MANIFEST> \
bash examples/tau_bench/run_tau_outcome.sh
```

Both commands default to 100 optimizer steps, a 4096-token response cap per training decision, save every 10 steps, and retain all checkpoints. Set `SMOKE=1` for one optimizer step with a two-decision trajectory cap. Generated Parquet data and checkpoints are placed under the run directory.

## Final Evaluation

Copy `models.example.tsv` to a local, ignored registry and replace the placeholder
paths. The optional third column is a verl `global_step_*` checkpoint; use `-`
for an ordinary Hugging Face model directory.

For the official full-domain evaluation, use the native Tau runner:

```bash
export OPENROUTER_API_KEY=<OPENROUTER_API_KEY>
PYTHON=<PYTHON_3_12> \
MODEL_SPECS_FILE=<MODEL_REGISTRY_TSV> \
RUN_DIR=runs/tau_native_eval_final \
bash examples/tau_bench/run_tau_native_eval.sh
```

The default `AGENT_PROTOCOL=strict_native` delegates reasoning and tool-call
parsing to vLLM. To evaluate with the same prompt and raw action parser used by
training, use a separate result directory:

```bash
AGENT_PROTOCOL=training_compatible \
TRAINING_DECISION_LIMIT=30 \
TRAINING_INVALID_ACTION_LIMIT=10 \
MODEL_SPECS_FILE=<MODEL_REGISTRY_TSV> \
RUN_DIR=runs/tau_training_compatible_eval \
bash examples/tau_bench/run_tau_native_eval.sh
```

In `training_compatible` mode, vLLM returns unparsed Qwen output and the adapter
applies the training `parse_action` and schema validation code. Invalid outputs
consume the decision budget and are resampled from the unchanged environment
state without entering conversation history. Tau still provides the official
tasks, environment, user simulator, and deterministic reward components. Before
native evaluation, the driver removes only `NL_ASSERTION` from each task copy's
reward basis. Keep strict native and training-compatible results in distinct run
directories.

This evaluates the complete official Airline `base` (50 tasks), Retail `base`
(114 tasks), and Telecom `base` (114 tasks) sets, with three trials per task.
The student is served locally by eight TP=1 vLLM replicas. Only the user
simulator uses OpenRouter, with temperature zero and reasoning disabled.
Scoring removes only `NL_ASSERTION` from each task copy's reward basis, then
delegates all remaining DB/ENV/ACTION/COMMUNICATE components to Tau's native
`EvaluationType.ALL`; LLM review and hallucination judging are disabled.

Native results are checkpointed in 100-task shards under `RUN_DIR`. This avoids
large repeated JSON rewrites on Telecom and allows an interrupted run to resume
completed trials automatically. A run created before the deterministic
NL-exclusion fix can be resumed once with
`ALLOW_NL_ASSERTION_PROTOCOL_UPGRADE=1`; migration is accepted only when every
other protocol field matches and saves the previous protocol as
`protocol.env.pre_nl_fix_v2`. `summary.json` and `summary.csv` report raw
success rate and native pass-hat-k metrics. Use `NUM_TASKS=1 DOMAINS=airline`
for a smoke run.

The older VERL validation path remains available for the fixed held-out split
used during training:

```bash
PYTHON=<TRAINING_PYTHON> \
MODEL_SPECS_FILE=<MODEL_REGISTRY_TSV> \
QUALIFICATION_MANIFEST=<QUALIFICATION_MANIFEST> \
bash examples/tau_bench/run_tau_eval.sh
```

It covers the 20 Airline and 40 Retail validation tasks at seeds 300, 301, 302,
and 303. Completed model/seed jobs are skipped on resume.
