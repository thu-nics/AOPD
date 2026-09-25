# Fixed training data

The default main recipe uses a fixed pool of **5,942 AWM tasks** (794
environments) and **755 EnvScaler tasks** (49 environments). It does not apply
an expert-success qualification gate. At each step a deterministic schedule
selects 59 AWM and 5 EnvScaler tasks. Tau uses its official training tasks;
Self-AOPD S2 additionally requires 178 reviewed training-only customer briefs.

Data and audit artifacts are not Git source. The expected local layout is:

```text
data/frozen/
  bundle.json
  awm/02_deterministic_audit/
  awm/03_static_feasibility_judge/
  envscaler/01_deterministic_audit/
  envscaler/02_static_feasibility_judge/
  tau/customer_briefs.json
```

Verify a supplied bundle before launching:

```bash
mkdir -p data
tar -xzf /path/to/aopd-data-v1.tar.gz -C data
python -m aopd.data verify data/frozen
```

The prepared archive SHA-256 is
`c1bab1483c7982e2f299a6adf119a74046d03c6af09cd2fa7cc0b42cedfc0b13`.

Verification checks every bundled file's SHA-256, native pool membership,
manifest protocol, deterministic mixed scheduling and reviewed brief evidence.
The complete 14-file identity is pinned in source, not trusted from the archive
alone. `bundle` and `verify` identify this exact release; rebuilt/custom pools
must be selected explicitly through runtime data paths.
The training Parquet hashes are:

| Pool | SHA-256 |
|---|---|
| AWM | `86e7580a7b41459a805c41b6212babea10a4ce41462e9586466f6d173beceb46` |
| EnvScaler | `464c6d1f890aa447e2ec2a20d5b6b53dc0f608fd2914cf091fb87968f7b49a4f` |

Maintainers can package existing audit artifacts without editing the originals:

```bash
python -m aopd.data bundle --research-runs /path/to/research/runs \
  --briefs /path/to/reviewed/customer_briefs.json --output data/frozen
```

The bundle removes source-machine absolute paths from copied provenance fields
and refreshes dependent hashes. It never modifies task rows. No historical
teacher, matcher or judge caches are included.

## Optional rebuild

The existing data adapters retain deterministic auditing and the single-task
static feasibility judge. AWM additionally retains its tokenizer-based context
selection. Use the module CLI help to supply a fresh output directory; API
screening is opt-in and incurs cost. A rebuilt pool can differ when the remote
judge changes, so use the frozen pool for comparable experiments.
Context-selection counts also depend on the tokenizer and prompt budget; a
custom rebuild uses its own audit counts, not the historical eligible count.
Source revision/hash, complete task membership and output integrity checks
remain mandatory. This does not loosen frozen-bundle verification.

EnvScaler stages:

```bash
python -m agent_system.environments.env_package.envscaler.filtering \
  --source-root "$ENVSCALER_SOURCE" --output-dir runs/data/envscaler/deterministic
python -m agent_system.environments.env_package.envscaler.screening \
  --source-root "$ENVSCALER_SOURCE" --deterministic-dir runs/data/envscaler/deterministic \
  --output-dir runs/data/envscaler/static --api-key-env DEEPSEEK_API_KEY
```

AWM source preparation downloads the pinned dataset revision when needed:

```bash
python -m agent_system.environments.env_package.awm.data.prepare \
  --data-dir "$AWM_DATA" --output-dir runs/data/awm/raw
```

The remaining AWM modules are `agent_system.environments.env_package.awm.data.selection`,
`.deterministic_health` and `.feasibility`, in that order. Each provides `--help`.
Context selection needs a running AWM server and the student's tokenizer;
feasibility uses reset observations and static implementation/verifier evidence,
not a full expert rollout. Its `--candidate-manifest` must be the selection
manifest associated with the input data, not the final health manifest.

Supply the data archive separately; it is not included in Git. Keep the
[third-party attribution](third_party.md) with any distributed archive.
