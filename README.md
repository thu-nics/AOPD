# Agentic On-Policy Distillation

AOPD trains tool-using agents from teacher actions at student-visited states,
without requiring teacher token probabilities. This release builds on
[verl-agent](https://github.com/langfengq/verl-agent) and veRL.

Included: AWM + EnvScaler AOPD training, Tau AOPD training, a separate Tau
outcome-GRPO baseline, and checkpoint export. Benchmark evaluation runners and KD/OPD
implementations are external and are not installed by this repository.

## Start here

1. Follow [installation](docs/installation.md). For AWM + EnvScaler, also obtain
   the fixed [data bundle](docs/data.md); Tau uses its source repository's official tasks.
2. Copy one runtime template to `configs/runtime.local.yaml`. Set model/source
   paths, services and GPUs. Keep API keys in environment variables, not YAML.
3. Inspect the launch plan, then run a bounded smoke before training:

```bash
python -m aopd train tau --runtime configs/runtime.local.yaml --check
python -m aopd train tau --runtime configs/runtime.local.yaml --smoke
python -m aopd train tau --runtime configs/runtime.local.yaml
```

Templates: [Tau](configs/runtime.tau.example.yaml),
[AWM + EnvScaler](configs/runtime.main.example.yaml).
The Tau template expects a Qwen3-32B teacher and Qwen3.8-27B auxiliary service;
model names and endpoints are supplied by the user. See [runtime configuration](docs/runtime.md) for
local/API services, arbitrary GPU placement, and shared roles.

## Commands

| Task | Command |
|---|---|
| Main training | `python -m aopd train main --runtime configs/runtime.local.yaml` |
| Tau AOPD training | `python -m aopd train tau --runtime configs/runtime.local.yaml` |
| Tau outcome-GRPO | See the [standalone command](docs/runtime.md#tau-outcome-grpo) |
| Resume training | Append `--resume /path/to/ckpt/global_step_N` |
| Periodic validation | Configure `validation` in runtime YAML; disabled by default |
| Export checkpoint | `python -m aopd export --checkpoint /path/to/global_step_N --output /path/to/new-hf-model` |
| Inspect training without running | Append `--check` to a train command |
| TensorBoard | `tensorboard --logdir runs` |
| Tests | `python -m pytest tests/release tests/tau_bench tests/awm tests/envscaler -q` |

Each invocation writes to `runs/<UTC timestamp>-<recipe>/`, including resolved
configuration, logs, caches, metrics and checkpoints. `--run-dir` overrides it.
The launcher refuses occupied GPUs and never kills unrelated jobs.

## Recipe reference

- [Protocols and recipes](docs/protocols.md): rewards, budgets and splits.
- [Fixed training data](docs/data.md): pool layout and data preparation.

## Code map and attribution

`aopd/` contains launch/configuration utilities. `agent_system/` contains
environment adapters and state-group rollout. `gigpo/` contains advantage
estimation; `verl/` contains training, workers and checkpointing. Scientific
defaults live in `configs/recipes/` and the underlying Hydra configs.

Upstream copyright notices are retained. See [LICENSE](LICENSE) and
[third-party notices](docs/third_party.md). Models and benchmark data retain
their own licenses.
