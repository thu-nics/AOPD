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
- The system message, task, and initial scaffolded `list_tools` exchange are
  pinned. At most three complete recent action/result exchanges are retained.
  The exact budget-trimmed chat is shared by all four candidates and the
  teacher.
- The scaffold calls `list_tools` after reset. It is not a student decision,
  teacher query, reward, or loss row. A later student `list_tools` call is a
  masked meta action.
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
  pure-code verifier. SQL+LLM judging is a separate optional protocol.

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
export PYTHON=/opt/venvs/verl-agent-sokoban/bin/python
bash examples/awm/install_awm.sh
```

The installer refuses to mutate an existing OpenEnv checkout at another commit.
Network commands honor the standard proxy variables from the shell.

Prepare the complete pinned dataset and deterministic ID-only subsets:

```bash
$PYTHON examples/awm/prepare_data.py \
  --data-dir "$HOME/.cache/openenv/awm" \
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

## Start AWM

AWM scenario code is trusted research code and is not isolated inside the server
process beyond the outer machine/container boundary. Run it only on a suitable
development host.

```bash
export OPENENV_ROOT=/opt/src/openenv-awm
export AWM_DATA_DIR="$HOME/.cache/openenv/awm"
bash examples/awm/start_server.sh
```

The launchers expect `http://127.0.0.1:8000/stats` to be healthy.

## Train

Run semantic training from the `deepseek_api` tmux shell so the pane-local
`DEEPSEEK_API_KEY` is inherited:

```bash
MODEL_PATH=/mnt/public2/yuanhuining/models/Qwen3-4B \
  bash examples/awm/run_semantic.sh
```

The normal launcher defaults to `TRAIN_SPLIT=all` and derives 1,250 optimizer
steps at batch size 8, so its deterministic sequential sampler covers all 10,000
tasks exactly once. `TRAIN_STEPS` remains an explicit override. Set
`TRAIN_SPLIT=dev` only for integration work or short development runs. Every launch
strictly verifies the fixed source hashes, manifest, and ordered Parquet task IDs.

One-step development smoke:

```bash
SMOKE=1 MODEL_PATH=/mnt/public2/yuanhuining/models/Qwen3-4B \
  bash examples/awm/run_semantic.sh
```

The isolated outcome baseline has no DeepSeek dependency:

```bash
MODEL_PATH=/mnt/public2/yuanhuining/models/Qwen3-4B \
  bash examples/awm/run_outcome.sh
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

`eval_awm.py` uses `AWMEnv` and its native `reset`, `list_tools`, `step`,
`verify`, and `done` methods directly. It never creates the training Ray rollout
stack. `run_eval.sh` starts one OpenAI-compatible vLLM server, keeps the model
resident while every selected task runs, and stops only that server process at
the end:

```bash
SPLIT=smoke MODEL_PATH=/mnt/public2/yuanhuining/models/Qwen3-4B \
  bash examples/awm/run_eval.sh
```

Set `START_VLLM=0 API_BASE=http://host:port/v1` to use an already persistent
server. `SEED` defaults to 300 and is part of every result and the strict run
identity. Evaluation incrementally writes raw per-task trajectories and verifier
results before producing a summary. `--resume` is accepted only when the
complete protocol identity matches.
For local checkpoints, that identity hashes the contents of every artifact file,
including all weight shards.
