# Agentic OPD training

This directory contains cross-environment Agentic OPD launchers. Environment-
specific installation, data processing, runtime adapters, and standalone
evaluation remain under their respective `examples/awm`, `examples/envscaler`,
and `examples/tau_bench` directories.

- `run_mixed_agentic_opd.sh` launches the main AWM plus EnvScaler training
  recipe with periodic Tau validation.
- `run_mixed_agentic_opd_smoke.sh` launches its one-step smoke configuration.
- `run_mixed_qwen37_flash_teacher.sh` uses DashScope `qwen3.7-flash` as the
  K=3 teacher while retaining the configured DeepSeek matcher and judges.

These entry points delegate to the shared AWM training launcher and accept the
same environment-variable and Hydra overrides. Prepared AWM and EnvScaler
healthy-pool manifests remain required for non-smoke training.

## Qwen3.7-Flash teacher

The Qwen launcher uses native DashScope tool calls with thinking enabled,
`parallel_tool_calls=false`, temperature `0.6`, top-p `0.95`, a 4,096-token
thinking budget, and an 8,192-token response ceiling. The teacher cache remains
strictly scoped by provider, model, decoding parameters, and prompt protocol.

```bash
export DASHSCOPE_API_KEY=...
export DEEPSEEK_API_KEY=...  # matcher, runtime/terminal judges, and API user simulators
bash examples/agentic_opd/run_mixed_qwen37_flash_teacher.sh
```

Override `ORACLE_API_BASE` when using a workspace-specific DashScope endpoint.
