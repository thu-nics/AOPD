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

The protocol uses raw completion for Base-model compatibility. The prompt includes
format-only examples, requests an AIME-style final `\boxed{ACTION}` containing
plain action text, and ends with a `Response:` cue. It strictly projects the
extracted action onto the current admissible actions and does not configure a
format stop. The response limit is 16K tokens and the model context is 32K.
Sampling uses `temperature=0.6`, `top_p=0.95`, and `top_k=20`.

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
