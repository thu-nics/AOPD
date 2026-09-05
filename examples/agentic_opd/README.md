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

AWM and EnvScaler use the same external-source lifecycle: each resolves its checkout
from a repository sibling by default, accepts an explicit root override, and has a
dedicated setup script that creates a missing checkout at a pinned commit. Training,
filtering, and runtime entry points never clone implicitly; they fail with the relevant
setup command when the checkout is absent. Both reject commit drift and tracked source
modifications while allowing untracked runtime caches such as `__pycache__`.

Generated Parquet pools, manifests containing run-specific paths, teacher
caches, and rollout artifacts remain under `runs/` and should not be committed.
Git should contain only the builders, split/quota definitions, schemas, and
small example manifests needed to reproduce those artifacts.
Formal runs still require the prepared AWM and EnvScaler pool manifests. Before
launching, set `MODEL_PATH` and `TAU_USER_LLM` explicitly; current cluster values
are listed in `docs/temp_docs/agentic_opd_docs/cluster_runtime.md`.

## Qwen3.7-Flash teacher

The Qwen configuration uses native DashScope tool calls with thinking enabled,
`parallel_tool_calls=false`, temperature `0.6`, top-p `0.95`, a 4,096-token
thinking budget, and an 8,192-token response ceiling. Matcher and both runtime
and terminal judges inherit the same Qwen provider/model/endpoint by default;
the matcher uses its deterministic non-thinking decoding protocol. Caches remain
strictly scoped by role, provider, model, decoding parameters, and prompt protocol.

```bash
export DASHSCOPE_API_KEY=...
export DEEPSEEK_API_KEY=...  # only needed by this example's Tau user simulator
MODEL_PATH=/path/to/student-model \
TAU_USER_LLM=deepseek/deepseek-v4-flash \
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

## GLM-5.3-Flash teacher

The ZAI configuration uses native tool calls with mandatory thinking,
`reasoning_effort=max`, `parallel_tool_calls=false`, temperature `1.0`, top-p
`0.95`, and preserved interleaved reasoning (`clear_thinking=false`). GLM is
also used for the frozen matcher and runtime/terminal judges by default. Because
GLM cannot disable thinking, its matcher uses the same mandatory-thinking API
mode while retaining the strict pairwise JSON-equivalence prompt. No
GLM-specific launcher is required.

Teacher votes are independent state queries: cached GLM reasoning is retained
for audit but is not replayed into later student states.

```bash
export ZAI_API_KEY=...
export DEEPSEEK_API_KEY=...  # only needed by this example's Tau user simulator
MODEL_PATH=/path/to/student-model \
TAU_USER_LLM=deepseek/deepseek-v4-flash \
ORACLE_PROVIDER=zai \
ORACLE_MODEL=glm-5.3-flash \
ORACLE_API_BASE=https://open.bigmodel.cn/api/paas/v4 \
ORACLE_API_KEY_ENV=ZAI_API_KEY \
ORACLE_ENABLE_THINKING=true \
ORACLE_REASONING_EFFORT=max \
ORACLE_TEMPERATURE=1.0 \
ORACLE_TOP_P=0.95 \
ORACLE_MAX_TOKENS=8192 \
bash examples/agentic_opd/run_mixed_agentic_opd.sh
```

Teacher, matcher, and runtime-judge caches are scoped by provider, model,
decoding parameters, and prompt protocol, so records cannot cross provider
boundaries. Setting the four `ORACLE_{PROVIDER,MODEL,API_BASE,API_KEY_ENV}`
identity variables is sufficient; each auxiliary role may still be overridden
explicitly. The generic launcher supports `deepseek`, `dashscope`, and `zai`.
