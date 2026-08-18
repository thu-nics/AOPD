# Agentic OPD training

This directory contains cross-environment Agentic OPD launchers. Environment-
specific installation, data processing, runtime adapters, and standalone
evaluation remain under their respective `examples/awm`, `examples/envscaler`,
and `examples/tau_bench` directories.

- `run_mixed_agentic_opd.sh` launches the main AWM plus EnvScaler training
  recipe with periodic Tau validation.
- `run_mixed_agentic_opd_smoke.sh` launches its one-step smoke configuration.

Both entry points delegate to the shared AWM training launcher and accept the
same environment-variable and Hydra overrides. Prepared AWM and EnvScaler
healthy-pool manifests remain required for non-smoke training.
