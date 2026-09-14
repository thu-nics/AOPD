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
history, tool-call IDs, or repetition keys. Message equivalence is a separate
path, described below.

## Message equivalence

The shared Tau/AWM/EnvScaler prompt checks constraints **before** accepting
paraphrases. Requests for information/authorization, conditional plans and reports
of execution are different conversational steps. A completion or handoff
announcement cannot replace an information request or offer. Text describing a
tool call is not an executed call. Essential answers, prerequisites, operation
scope, facts, quantities, commitments and protocol-required literal text must
remain intact; a shared goal cannot override these constraints.

Within these boundaries, concise summaries, optional grounded recaps and extra
relevant clarification may match. For example, asking for an order ID may match
asking for that ID plus which item needs help. Each candidate/teacher pair is
judged independently. This adds no model pass, intent-extraction stage,
tool-specific gate, or hidden task evidence.

These are matcher instructions, not a deterministic execution-validity gate.
Exact equality still bypasses the API. Two equivalent but policy-wrong messages
can still match: this is an equivalence matcher, not a task-success judge. The
prompt alone does not guarantee that every awarded completion claim followed
an actual tool execution.

Tau and AWM/EnvScaler message protocol **5** use one `{"equivalent": bool}`
request per unique non-exact pair, with parallel requests, single-flight and
persistent pair caching. The shared DeepSeek preset for **messages and tool
arguments** is thinking enabled / reasoning effort `low` / max tokens `32768`;
temperature and top-p are omitted. Matcher concurrency defaults to 32,
independent of teacher concurrency. Non-DeepSeek defaults are preserved:
Tau's local Qwen message matcher is thinking / temperature 0 / 8192 tokens,
and its tool matcher remains non-thinking / 1024 tokens.

Identity, prompt, decoding and evidence partition cached decisions. The new
DeepSeek configuration does not reuse older message or tool verdicts, but
teacher and runtime-judge cache identities are unchanged. Malformed/truncated
matcher replies are missing supervision, never a false equivalence verdict.

### Tau training-only transfer guard

`env.tau.transfer_reward_guard_enabled=true` caps the final candidate reward at
**0** for the exact native fixed handoff notice (case/whitespace normalized)
when no successful `transfer_to_human_agents` call exists in the executed
history. The success flag requires an assistant tool call with a linked native
`Transfer successful` result, persists across prompt truncation, and resets
per task. Offers, questions, conditional/quoted notices and other paraphrases
are not heuristically classified by this narrow guard.

It is a separate reward rule, **not** an invalid action or matcher decision:
raw semantic reward, match counts/matrix and K votes are retained. Both final
reward and appearance-selection score become 0; the row remains trainable.
There is no special execution veto, trajectory mask or task quarantine. An
all-zero group follows the existing equal-reward skip. Native outcome/eval
and earlier valid state groups are unchanged. Monitor
`env/transfer_without_tool_candidate_rate` (padding excluded).

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
The tool-rule update does not alter teacher generation identity. Existing
teacher-cache compatibility/import checks still apply; message-cache changes
are described above.
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
