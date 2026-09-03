# Agentic OPD training

This directory contains cross-environment Agentic OPD launchers. Environment-
specific installation, data processing, runtime adapters, and standalone
evaluation remain under their respective `examples/awm`, `examples/envscaler`,
and `examples/tau_bench` directories.

- `run_mixed_agentic_opd.sh` launches the main AWM plus EnvScaler training
  recipe with periodic Tau validation.
- `run_mixed_agentic_opd_smoke.sh` launches its one-step smoke configuration.

These entry points delegate to the shared AWM training launcher and accept the
same environment-variable and Hydra overrides. `run_mixed_agentic_opd.sh`
is the provider-agnostic canonical entry point; the smoke file is a bounded
convenience preset. Model paths, GPU topology, domain quotas, and
every `ORACLE_*` setting remain overridable.

Generated Parquet pools, manifests containing run-specific paths, teacher
caches, and rollout artifacts remain under `runs/` and should not be committed.
Git should contain only the builders, split/quota definitions, schemas, and
small example manifests needed to reproduce those artifacts.
Formal runs still require the prepared AWM and EnvScaler pool manifests.

## Qwen3.7-Flash teacher

The Qwen configuration uses native DashScope tool calls with thinking enabled,
`parallel_tool_calls=false`, temperature `0.6`, top-p `0.95`, a 4,096-token
thinking budget, and an 8,192-token response ceiling. The teacher cache remains
strictly scoped by provider, model, decoding parameters, and prompt protocol.

```bash
export DASHSCOPE_API_KEY=...
export DEEPSEEK_API_KEY=...  # matcher, runtime/terminal judges, and API user simulators
ORACLE_PROVIDER=dashscope \
ORACLE_MODEL=qwen3.7-flash \
ORACLE_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1 \
ORACLE_API_KEY_ENV=DASHSCOPE_API_KEY \
ORACLE_ENABLE_THINKING=true \
ORACLE_REASONING_EFFORT=null \
ORACLE_THINKING_BUDGET=4096 \
ORACLE_TEMPERATURE=0.6 \
ORACLE_TOP_P=0.95 \
ORACLE_MAX_TOKENS=8192 \
bash examples/agentic_opd/run_mixed_agentic_opd.sh
```

All `ORACLE_*` values can be changed for another OpenAI-compatible teacher.
