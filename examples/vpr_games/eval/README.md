# Evaluation

## In-domain Games

`eval_in_domain_all.sh` evaluates the registered Sokoban, Sudoku, and
Minesweeper checkpoints. Results are written under `runs/` and can be resumed
by reusing `RUN_DIR`.

```bash
MODEL_PATH=/path/to/base-model \
RUN_DIR=runs/eval_in_domain \
  bash examples/vpr_games/eval/eval_in_domain_all.sh
```

Checkpoint paths and task/model filters can be overridden with the environment
variables declared at the top of the script.

## ALFWorld And WebShop

`eval_agentic_ood_all.sh` evaluates any model manifest on the complete
ALFWorld `valid_unseen` split and WebShop 500-task test split. The defaults run
ALFWorld with five sampling seeds and WebShop with three.

Each manifest row selects its prompt rendering explicitly: use `raw` for the
untrained Base model and `chatml` for zero-RL checkpoints. ChatML rendering uses
`enable_thinking=true`. Both benchmarks use their stock verl-agent prompt with
only the sentence requiring literal `<think>...</think>` tags removed. By
default, every model is evaluated once with `<action>...</action>` and once with
`\boxed{ACTION}`; the generated action is strictly projected onto the current
admissible actions and no format stop is configured.

The default protocol retains the two most recent environment turns, permits 50
ALFWorld steps and 30 WebShop steps, and uses a 16K prompt, 8K response-per-turn,
and 32K model context. Prompts that exceed the 16K budget are middle-truncated,
preserving the task prefix and the admissible-action/output-format suffix.
Sampling uses `temperature=0.6`, `top_p=0.95`, `top_k=20`, and `min_p=0`.

Create a runtime manifest from `agentic_ood_models.example.tsv`, then run:

```bash
ALFWORLD_DATA=/path/to/alfworld/data \
WEBSHOP_DATA_DIR=/path/to/webshop/data \
MODEL_MANIFEST=runs/agentic_ood_models.tsv \
RUN_DIR=runs/eval_agentic_ood \
  bash examples/vpr_games/eval/eval_agentic_ood_all.sh
```

Relative model paths are resolved from the repository root. VERL FSDP
checkpoints are merged once and cached under `runs/eval_model_cache/`.
Reusing the same `RUN_DIR` resumes completed model/task/seed jobs only when
the protocol, evaluator revision, and model identity still match.

The WebShop full Lucene index must exist at
`agent_system/environments/env_package/webshop/webshop/search_engine/indexes`.
Use the repository's WebShop `setup.sh -d all` flow before evaluation.

The evaluator never stops existing processes. By default it waits until every
GPU listed in `CUDA_VISIBLE_DEVICES` is free. Select genuinely idle GPUs or run
on another machine that shares the repository storage.
