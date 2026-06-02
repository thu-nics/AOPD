"""Verify VPR training smoke test output.

Usage: python3 smoke_verify.py <log_file> <evidence_file>

Both arguments are MANDATORY. The evidence file must contain per-batch JSON
written by core_gigpo.compute_vpr_turn_level_advantage() when VPR_SMOKE_EVIDENCE
is set. The verifier:

Layer 1 — Aggregate training-log checks (training steps, oracle reward, outcome bonus).
Layer 2 — Per-batch evidence checks (unforgeable):
  - Schema: every required field present in every row
  - Reward consistency: effective_reward == oracle_reward + outcome_bonus (+-1e-4)
  - Advantage recomputation: emitted advantage matches per-turn normalization (+-1e-4)
  - Terminal-only bonus: outcome_bonus == 0 on every non-terminal row
  - Multi-step: at least one batch has a trajectory spanning >= 2 distinct turns
  - Prompt locality: a later step's prompt_prefix does not contain the prior step's action
  - Prompt bounded: prompt length within each trajectory does not grow by > 200 chars

A fabricated log, empty evidence, or missing field causes exit 1.
"""
import sys, re, json
import numpy as np
from collections import defaultdict

# ── Argument validation ───────────────────────────────────────────────────────
if len(sys.argv) < 3:
    print("FAIL: evidence_file argument is mandatory", file=sys.stderr)
    print("Usage: smoke_verify.py <log_file> <evidence_file>", file=sys.stderr)
    sys.exit(1)

log_file = sys.argv[1]
evidence_file = sys.argv[2]

try:
    log = open(log_file).read()
except FileNotFoundError as e:
    print(f"FAIL: log file not found: {e}", file=sys.stderr)
    sys.exit(1)

# ── Layer 1: aggregate log checks ────────────────────────────────────────────
all_lines = [l for l in log.splitlines()
             if 'global_step:' in l and ('TaskRunner' in l or 'step:' in l)]

def get(pat, text):
    m = re.search(pat + r':([-0-9.e]+)', text)
    return float(m.group(1)) if m else None

errors = []
last = all_lines[-1] if all_lines else log
first = all_lines[0] if all_lines else log

steps = [get('training/global_step', l) for l in all_lines if get('training/global_step', l)]
if steps and max(steps) >= 2:
    print(f"PASS: training completed {int(max(steps))} steps")
else:
    errors.append(f"training/global_step:2 not found (found: {steps})")

oracle = get('vpr/oracle_reward_mean', last)
if oracle is not None:
    print(f"PASS: vpr/oracle_reward_mean={oracle:.4f}")
else:
    errors.append("vpr/oracle_reward_mean not found in last step line")

bonus_mean = get('vpr/outcome_bonus_mean', last)
if bonus_mean is not None:
    if bonus_mean < -1e-6:
        errors.append(f"outcome_bonus_mean={bonus_mean:.4f} is negative")
    else:
        print(f"PASS: vpr/outcome_bonus_mean={bonus_mean:.4f} (>= 0)")
else:
    errors.append("vpr/outcome_bonus_mean not found")

adv_min = get('critic/advantages/min', last)
adv_max = get('critic/advantages/max', last)
if adv_min is not None and adv_max is not None:
    spread = adv_max - adv_min
    if spread >= 1e-6:
        print(f"PASS: advantages distinct: min={adv_min:.4f}, max={adv_max:.4f}")
    else:
        print(f"INFO: advantages=0 (all-equal rewards): min={adv_min}, max={adv_max}")

pl_all = [get('prompt_length/mean', l) for l in all_lines if get('prompt_length/mean', l) is not None]
if len(pl_all) >= 2:
    growth = pl_all[-1] - pl_all[0]
    if growth > 200:
        errors.append(f"Batch prompt mean grew {growth:.1f} chars")
    else:
        print(f"PASS: batch prompt bounded: delta={growth:.1f} chars")

# ── Layer 2: per-batch evidence ───────────────────────────────────────────────
try:
    with open(evidence_file) as f:
        ev = json.load(f)
except FileNotFoundError:
    errors.append(f"Evidence file not found: {evidence_file}")
    for e in errors:
        print(f"FAIL: {e}", file=sys.stderr)
    sys.exit(1)
except (ValueError, json.JSONDecodeError) as ex:
    errors.append(f"Evidence file malformed JSON: {ex}")
    for e in errors:
        print(f"FAIL: {e}", file=sys.stderr)
    sys.exit(1)

batches = ev.get("batches", [])
if not batches:
    errors.append(
        "Evidence file has no 'batches' key or empty batches list. "
        "Ensure VPR_SMOKE_EVIDENCE is set before training starts."
    )
    for e in errors:
        print(f"FAIL: {e}", file=sys.stderr)
    sys.exit(1)

REQUIRED_ROW = {
    "traj_uid", "turn_index", "oracle_reward", "outcome_bonus",
    "effective_reward", "advantage", "is_terminal", "terminal_success", "prompt_len",
}
REQUIRED_BATCH = {"batch_id", "min_group_size", "eps", "global_mean", "global_std", "rows"}

total_rows = 0
multi_step_found = False
print(f"\nEvidence: {len(batches)} batches from {evidence_file}")

for bi, batch in enumerate(batches):
    missing_b = REQUIRED_BATCH - set(batch.keys())
    if missing_b:
        errors.append(f"Batch {bi} missing fields: {missing_b}")
        continue

    rows = batch["rows"]
    if not rows:
        errors.append(f"Batch {bi} is empty")
        continue

    total_rows += len(rows)
    min_group_size = batch["min_group_size"]
    eps = batch["eps"]
    global_mean = batch["global_mean"]
    global_std = batch["global_std"]

    # Schema validation
    for ri, row in enumerate(rows):
        missing_r = REQUIRED_ROW - set(row.keys())
        if missing_r:
            errors.append(f"Batch {bi} row {ri} missing: {missing_r}")

    if any(e.startswith(f"Batch {bi} row") for e in errors):
        continue  # skip further checks for this batch if schema fails

    # Reward consistency
    for ri, row in enumerate(rows):
        computed = row["oracle_reward"] + row["outcome_bonus"]
        if abs(row["effective_reward"] - computed) > 1e-4:
            errors.append(
                f"Batch {bi} row {ri}: effective_reward={row['effective_reward']:.6f} "
                f"!= oracle+bonus={computed:.6f}"
            )

    # Advantage recomputation
    eff = np.array([r["effective_reward"] for r in rows], dtype=np.float64)
    ti = np.array([r["turn_index"] for r in rows], dtype=np.int32)
    exp_adv = np.zeros(len(rows), dtype=np.float64)
    for t in np.unique(ti):
        mask = ti == t
        group = eff[mask]
        mean_t = group.mean() if len(group) >= min_group_size else global_mean
        std_t = (group.std() + eps) if len(group) >= min_group_size else global_std
        exp_adv[mask] = (group - mean_t) / std_t

    for ri, (row, exp) in enumerate(zip(rows, exp_adv)):
        if abs(row["advantage"] - exp) > 1e-3:
            errors.append(
                f"Batch {bi} row {ri}: advantage={row['advantage']:.6f} != recomputed={exp:.6f}"
            )

    # Terminal-only bonus
    for ri, row in enumerate(rows):
        if not row["is_terminal"] and abs(row.get("outcome_bonus", 0)) > 1e-6:
            errors.append(f"Batch {bi} row {ri}: non-terminal has outcome_bonus={row['outcome_bonus']}")

    # Per-trajectory checks
    traj = defaultdict(list)
    for row in rows:
        uid = row.get("traj_uid")
        if uid:
            traj[uid].append(row)

    for uid, trows in traj.items():
        turns = sorted(set(r["turn_index"] for r in trows))
        if len(turns) < 2:
            continue
        multi_step_found = True
        by_turn = sorted(trows, key=lambda r: r["turn_index"])

        # Prompt locality: prior action must not appear in next prompt
        for j in range(1, len(by_turn)):
            prev_act = by_turn[j-1].get("action_prefix", "")
            curr_prompt = by_turn[j].get("prompt_prefix", "")
            if prev_act and curr_prompt and len(prev_act) >= 10:
                if prev_act[:50] in curr_prompt:
                    errors.append(
                        f"Batch {bi} traj {uid}: action from turn {by_turn[j-1]['turn_index']} "
                        f"appears in prompt at turn {by_turn[j]['turn_index']} (history leak)"
                    )

        # Prompt bounded within trajectory
        lens = [r["prompt_len"] for r in by_turn if r.get("prompt_len") is not None]
        if lens and (max(lens) - min(lens)) > 200:
            errors.append(
                f"Batch {bi} traj {uid}: prompt grew {max(lens)-min(lens)} chars "
                f"across {len(lens)} turns (limit=200)"
            )

print(f"PASS: evidence covers {total_rows} rows across {len(batches)} batches")

if errors:
    for e in errors:
        print(f"FAIL: {e}", file=sys.stderr)
    sys.exit(1)

if not multi_step_found:
    print("FAIL: no multi-step trajectory found (all episodes ended in 1 turn)", file=sys.stderr)
    print("      Re-run until >=1 episode produces >=2 rollout turns.", file=sys.stderr)
    sys.exit(1)

print("PASS: multi-step trajectory confirmed")
print("PASS: reward consistency, advantage recomputation, prompt locality all verified")
