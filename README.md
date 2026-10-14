<h1 align="center">ActionOPD:<br>Distilling Decisions, Not Tokens</h1>

<p align="center">
  Huining Yuan<sup>*</sup>, Xiangmin Yi<sup>*</sup>, Zelai Xu, Zixu Wang,<br>
  Xinlei Chen, Xiao-Ping Zhang<sup>†</sup>, Yu Wang<sup>†</sup>, Chao Yu<sup>†</sup><br>
  <strong>Tsinghua University</strong><br>
  <sup>*</sup> Equal contribution &nbsp; <sup>†</sup> Corresponding authors
</p>

<p align="center">
  <a href="https://thu-nics.github.io/ActionOPD/"><img src="https://img.shields.io/badge/Project_Page-447890?style=for-the-badge" alt="Project Page"></a>
  <!-- Paper URL intentionally left unset. Wrap this badge in a link when available. -->
  <img src="https://img.shields.io/badge/Paper-555555?style=for-the-badge" alt="Paper">
  <a href="https://huggingface.co/collections/nics-efc/actionopd"><img src="https://img.shields.io/badge/Models-94508A?style=for-the-badge" alt="Models"></a>
</p>

Official implementation of **ActionOPD, an action-level on-policy distillation method** for
language-model agents. ActionOPD rewards semantic agreement with black-box teachers at
student-encountered states, without requiring teacher token probabilities or
tokenizer alignment. The student learns through its own reasoning and responses
rather than reproducing teacher-written reasoning.

[Method](#method-overview) · [Results](#highlights) · [Quick start](#quick-start) · [Documentation](#documentation) · [Citation](#citation)

## Method overview

ActionOPD starts from **maximizing expected teacher-action likelihood**: the student
probability mass of responses that realize the teacher's semantic action. This
objective yields a REINFORCE-form gradient with semantic agreement as reward,
motivating the following training procedure:

1. **Sample at the same state.** Query the teacher for three action samples and
   generate four candidate responses from the student.
2. **Score action agreement.** Compare tool calls or user-facing messages, not
   reasoning text. Matched actions receive a base reward plus a smaller frequency
   bonus for additional teacher matches. Invalid actions are penalized.
3. **Advance once, learn from the group.** Execute a highest-reward action to
   continue one trajectory. Group-relative policy optimization trains all student
   responses in retained groups, including their reasoning and action tokens.

<p align="center">
  <a href="https://thu-nics.github.io/ActionOPD/#method">
    <img src="https://thu-nics.github.io/ActionOPD/assets/figures/main-overview.svg?v=actionopd" width="100%" alt="Comparison of offline distillation, token-level on-policy distillation, outcome-based reinforcement learning, and ActionOPD's action-level semantic supervision.">
  </a>
</p>

Training uses local semantic rewards rather than terminal outcome rewards.
Agreement measures consistency with the teacher, not guaranteed correctness.

## Highlights

After **100 training steps** on a curated pool of **6,697 AWM and EnvScaler tasks**,
all four student–teacher configurations improve the average across four
out-of-distribution benchmarks by **3.3–6.9 percentage points**. Larger students
achieve higher final averages, and DeepSeek-V4-Flash yields higher student averages
at both sizes, demonstrating effective cross-family distillation.

| Training teacher | Qwen3-4B | Qwen3-8B |
|---|---:|---:|
| Base (no distillation) | 30.8 | 35.7 |
| Qwen3.7-Flash | 35.9 | 39.0 |
| DeepSeek-V4-Flash | **37.7** | **40.0** |

Scores are four-benchmark averages on a 0–100 scale (higher is better), equally
weighting BFCL accuracy, ToolSandbox similarity, AppWorld mean Normal/Challenge
task completion, and τ²-bench mean domain pass¹ using the first four trials per
task. Gains are computed before rounding.

- **Competitive distillation:** On τ²-bench, ActionOPD achieves competitive single-trial
  success against offline distillation and sampled-token OPD using only teacher
  generations. Repeated-success reliability (pass⁴) is lower than both baselines.
- **Semantic supervision:** ActionOPD has higher average single-trial success than the
  validity-reward, binary-reward, and random-commit ablations on training and
  held-out tasks.
- **Self-ActionOPD:** A same-model teacher with privileged context provides useful
  action supervision, improving held-out performance while the student uses only
  public context.

See the [project page](https://thu-nics.github.io/ActionOPD/#results) for detailed
comparisons, behavioral examples, and Self-ActionOPD results. Here, pass⁴ measures
success across all four trials, not best-of-four success.

## Quick start

### Release scope

This repository provides **AWM + EnvScaler ActionOPD training**, **τ²-bench ActionOPD
training**, and **checkpoint export**. It also includes a separate Tau
outcome-GRPO training entry point, which is not a distillation baseline in the
paper. Benchmark evaluation runners and KD/OPD implementations are external and
are not installed by this repository. Self-ActionOPD and ablation launch recipes are
not exposed in this release.

### Installation and configuration

Training requires **Linux, Python 3.12, and NVIDIA GPUs** with a compatible CUDA
driver. Follow the [installation guide](docs/installation.md) for dependencies
and environment setup.

```bash
git clone https://github.com/thu-nics/ActionOPD.git
cd ActionOPD
```

1. Complete [installation](docs/installation.md). For AWM + EnvScaler, also obtain
   and verify the fixed [data bundle](docs/data.md). Tau uses its source
   repository's official tasks.
2. Copy the [Tau template](configs/runtime.tau.example.yaml) or
   [AWM + EnvScaler template](configs/runtime.main.example.yaml) to
   `configs/runtime.local.yaml`. Configure model/source paths, services, and GPUs.
   Keep API keys in environment variables, not YAML.
3. Inspect the launch plan, then run a bounded smoke test before training:

```bash
python -m actionopd train tau --runtime configs/runtime.local.yaml --check
python -m actionopd train tau --runtime configs/runtime.local.yaml --smoke
python -m actionopd train tau --runtime configs/runtime.local.yaml
```

The example above runs Tau ActionOPD. Use `train main` with the main runtime template
for AWM + EnvScaler training. The Tau template expects a Qwen3-32B teacher and a
Qwen3.8-27B auxiliary service. Model names and endpoints are supplied by the user.
See [runtime configuration](docs/runtime.md) for local/API services, GPU placement,
and shared roles.

## Commands

| Task | Command |
|---|---|
| Main training | `python -m actionopd train main --runtime configs/runtime.local.yaml` |
| Tau ActionOPD training | `python -m actionopd train tau --runtime configs/runtime.local.yaml` |
| Tau outcome-GRPO | See the [standalone command](docs/runtime.md#tau-outcome-grpo) |
| Resume training | Append `--resume /path/to/ckpt/global_step_N` |
| Periodic validation | Configure `validation` in runtime YAML; disabled by default |
| Export checkpoint | `python -m actionopd export --checkpoint /path/to/global_step_N --output /path/to/new-hf-model` |
| Inspect training without running | Append `--check` to a train command |
| TensorBoard | `tensorboard --logdir runs` |
| Tests | `python -m pytest tests/release tests/tau_bench tests/awm tests/envscaler -q` |

Each invocation writes to `runs/<UTC timestamp>-<recipe>/`, including resolved
configuration, logs, caches, metrics and checkpoints. `--run-dir` overrides it.
The launcher refuses occupied GPUs and never kills unrelated jobs.

## Documentation

| Guide | Contents |
|---|---|
| [Installation](docs/installation.md) | Training dependencies and environment sources |
| [Runtime configuration](docs/runtime.md) | Model services, GPU placement, validation, and launch options |
| [Training data](docs/data.md) | Fixed AWM/EnvScaler pools, verification, and preparation |
| [Protocols and recipes](docs/protocols.md) | Rewards, training budgets, and task splits |

## Code structure

`actionopd/` contains launch/configuration utilities. `agent_system/` contains
environment adapters and state-group rollout. `gigpo/` contains advantage
estimation; `verl/` contains training, workers and checkpointing. Scientific
defaults live in `configs/recipes/` and the underlying Hydra configs.

## Citation

If you find ActionOPD useful in your research, please cite:

```bibtex
@misc{yuan2026actionopd,
  title  = {ActionOPD: Distilling Decisions, Not Tokens},
  author = {Yuan, Huining and Yi, Xiangmin and Xu, Zelai and Wang, Zixu
            and Chen, Xinlei and Zhang, Xiao-Ping and Wang, Yu and Yu, Chao},
  year   = {2026},
  note   = {Preprint}
}
```

## Acknowledgments and license

This implementation builds on [verl-agent](https://github.com/langfengq/verl-agent)
and [veRL](https://github.com/volcengine/verl).

Upstream copyright notices are retained. See [LICENSE](LICENSE) and
[third-party notices](docs/third_party.md). Models and benchmark data retain
their own licenses.
