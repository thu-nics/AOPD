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

Qualification evaluates all 30 Airline and 74 Retail training tasks with four trials each. A task is admitted when it succeeds in at least three trials and no successful trial contains an illegal action. Training hard-fails unless at least 20 Airline and 50 Retail tasks qualify.

## Training Metrics

- `episode/env/protocol_reward` and `episode/env/success_rate` report the deterministic terminal task score described above.
- `episode/env/valid_action_rate` reports schema-valid tool calls or non-empty user messages.
- `episode/env/oracle_hit_rate` reports the fraction of VPR committed actions that match the sampled oracle set; it is zero for outcome training.
- In VPR, `episode/reward` is the accumulated process reward of committed actions and is not a terminal task-success metric.
- Equal-reward VPR state groups are masked from the policy loss. Outcome DAPO resamples equal terminal-score groups up to the configured generation limit and always caps the final batch at the requested number of complete rollout groups.

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

Both commands default to 100 optimizer steps and save every 25 steps. Set `SMOKE=1` for one optimizer step with a two-decision trajectory cap. Generated Parquet data and checkpoints are placed under the run directory.

## Final Evaluation

Copy `models.example.tsv` to a local, ignored registry and replace the placeholder paths. The optional third column is a verl `global_step_*` checkpoint; use `-` for an ordinary Hugging Face model directory.

```bash
PYTHON=<TRAINING_PYTHON> \
MODEL_SPECS_FILE=<MODEL_REGISTRY_TSV> \
QUALIFICATION_MANIFEST=<QUALIFICATION_MANIFEST> \
bash examples/tau_bench/run_tau_eval.sh
```

The default evaluation covers all 20 Airline and 40 Retail test tasks at seeds 300, 301, 302, and 303. Completed model/seed jobs are skipped on resume. Raw generations, per-seed metrics, `summary.json`, and `summary.csv` are written under `RUN_DIR`.
