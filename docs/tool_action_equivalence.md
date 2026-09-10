# Tau / AWM / EnvScaler tool-action equivalence

The training reward path is shared:

```text
native action validation
  -> same tool + exact arguments
  -> source-scoped deterministic equivalence
  -> source-aware pairwise matcher for remaining differences
  -> restore every original teacher vote and sum Booleans
```

Different tool names never match. Invalid actions remain invalid. Comparisons
never execute extra candidates or modify execution arguments, teacher samples,
history, tool-call IDs, or repetition keys. Message matching is unchanged.

## Deterministic rules

Native Python/Pydantic defaults may be filled in a comparison-only copy when
execution proves that omission uses that default. JSON Schema annotations alone
are not proof. Additional rules in
`agent_system/environments/tool_matching_rules.json` require an exact family,
environment, public tool name, native function hash and complete module hash.

| Native operation | Comparison-only rule | Important boundary |
| --- | --- | --- |
| Tau transfer | Ignore summary | Still validate and execute the original summary |
| Tau retail return | Sort item-ID multiset | Preserve duplicate items |
| Tau retail exchange | Sort old/new pairs | Preserve mapping and duplicate pairs |
| EnvScaler env_142_rl appointment status | Native lowercase normalization | Not a global status/case rule |
| EnvScaler env_160_rl allowed ingredients | Native add/remove sets; None = empty | Preserve user and add/remove roles |
| AWM social_media_4 hidden subreddits | Native add/remove sets; None = empty | Do not apply to replacement APIs storing ordered CSV |

These cases have offline native-function counterfactual tests. In particular,
Tau return/exchange tests compare both complete cloned DBs and observations.
Other tools that use sorting internally are **not automatically normalized**:
sorting may apply only to outputs, and paired inputs or duplicate items can
change behavior.

## Matcher contract

There is no prose-field whitelist. Remaining same-tool differences—including
arrays, missing/null values, IDs, fixed strings, code and query strings—are
submitted with the actual schema, complete arguments, differing paths, public
history and source evidence. Evidence includes the relevant native function,
local helpers and request/data types. It excludes live DB snapshots, hidden task
answers, task verifiers and environment initialization.

The matcher judges the same material operation/information, not quality,
necessity, reward, repeated use or likely success. Identifiers, quantities,
literal-copy instructions, runtime enums and pair mappings cannot be ignored
merely because both actions seem useful—or both fail. Genuinely ambiguous
equivalence returns false.

Missing/ambiguous function identity, unresolved method dependencies, evidence
overflow and API/JSON failures are matcher infrastructure failures: mask the
current state group and end its owning trajectory, preserving previous valid
groups. Exact matches do not need source or an API request.

The explicit FastAPI `operation_id` takes precedence over same-named private
helpers. Source evidence is never silently truncated. Current budgets are
48,000 source characters and 120,000 characters for the complete pair evidence;
an API may impose a tighter token limit and then follow the same failure path.

## Caches and metrics

Tool matcher protocol **3** invalidates older tool verdicts. Keys include source,
rules, schema, public context, provider/model/endpoint and decoding settings.
Message matcher caches and teacher generation identity are unchanged by this
update. Existing teacher-cache compatibility/import checks still apply.
Duplicate teacher votes are retained; single-flight/persistent pair caching
deduplicates API work only. Fixed K reward scaling and equal-reward group masks
are unchanged.

`tool_argument_normalized_match_count` counts candidate × teacher-vote matches
rescued by deterministic normalization, excluding padding. It appears under the
existing teacher-selection diagnostic prefixes. Existing per-row
`tool_argument_semantic_match_count` records positive API matches.

Continuing a previous checkpoint is a **reward-protocol intervention**, not a
lossless continuation of the old experiment. Native outcome eval is unchanged.
AWM/EnvScaler can disable the entire tool-argument fallback with
`env.awm.oracle.tool_argument_matcher_enabled=false` for the exact baseline.

## Reproducing the audit

The audit reads native source and optional pool membership without running tools,
calling APIs or changing task pools:

```bash
python -m agent_system.environments.audit_tool_matching \
  --awm-pool runs/awm_data_processing/03_static_feasibility_judge/awm_training_pool.parquet \
  --envscaler-pool runs/envscaler_data_processing/02_static_feasibility_judge/envscaler_training_pool.parquet \
  --output runs/tool_matching_audit/inventory.json
```

Source directories default to sibling `openenv-awm-cache` and `EnvScaler`;
override with `--awm-data-dir` and `--envscaler-root`. Omitting a pool audits
all available source environments. AWM inventory names come from endpoint
declarations, so this report is not a live MCP-schema coverage test.

The 2026-09-10 healthy-pool inventory covers 794 AWM environments / 27,235
endpoint declarations and 49 EnvScaler environments / 878 tools. Native Tau
Airline/Retail coverage separately checks 14 / 16 tools. Pattern counts in the
inventory are review leads, **not counts of proven bugs**. The implemented
source-aware fallback covers the long tail; its statistical accuracy still
requires real-provider validation and cannot be proved by unit tests.

```bash
python -m pytest tests/test_action_matching.py tests/test_tool_matching_metadata.py \
  tests/awm tests/envscaler tests/tau_bench -q
```
