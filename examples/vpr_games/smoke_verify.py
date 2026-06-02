"""Verify VPR training smoke test output.

Usage: python3 smoke_verify.py <log_file> <evidence_file>

Both arguments are MANDATORY. The evidence file must contain per-batch JSON
written by core_gigpo.compute_vpr_turn_level_advantage() when VPR_SMOKE_EVIDENCE
is set. The verifier treats only oracle_reward, turn_index, is_terminal,
terminal_success, traj_uid, and the captured prompt/action text as inputs, and
independently recomputes everything else (outcome bonus, effective reward, batch
statistics, advantages) from the fixed training contract. This makes fabricated
metadata unable to pass.

Layer 1 — Aggregate training-log checks (training steps, oracle reward, outcome bonus).
Layer 2 — Per-batch evidence checks (unforgeable):
  - Schema: every required field present in every row
  - Finite numerics: no NaN/inf in any numeric field
  - Normalization contract: min_group_size == 4, eps == 1e-8
  - Outcome bonus: recomputed from is_terminal & terminal_success (success -> +1.0,
    else 0.0); emitted bonus must match (catches both spurious and missing bonuses)
  - Reward consistency: effective_reward == oracle_reward + recomputed bonus
  - Advantage recomputation: emitted advantage matches per-turn normalization computed
    from recomputed effective rewards and row-derived batch statistics
  - Multi-step: at least one trajectory spans >= 2 distinct turns
  - Per-episode reward diversity: at least one multi-step trajectory carries >= 2
    distinct non-zero per-step oracle rewards
  - Prompt locality: no earlier action (full text or extracted final <action> tag)
    appears in any later prompt unless it is part of the trajectory's static baseline
  - Prompt bounded: prompt token length within each trajectory does not grow by > 200

A fabricated log, empty evidence, or any failed check causes exit 1.
"""
import sys, re, json, math
import numpy as np
from collections import defaultdict

# Fixed smoke training contract — the verifier enforces these independently rather
# than trusting whatever the evidence emitted.
CONTRACT_MIN_GROUP_SIZE = 4
CONTRACT_EPS = 1e-8
OUTCOME_REWARD_SCALE = 1.0

_ACTION_TAG_RE = re.compile(r"<action>.*?</action>", re.DOTALL | re.IGNORECASE)


def _final_action_tag(text):
    """Return the last full <action>...</action> substring of text, or None."""
    matches = _ACTION_TAG_RE.findall(text or "")
    return matches[-1] if matches else None

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
        errors.append(f"Batch prompt mean grew {growth:.1f} tokens")
    else:
        print(f"PASS: batch prompt bounded: delta={growth:.1f} tokens")

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
    "prompt_prefix", "action_prefix",
}
REQUIRED_BATCH = {"batch_id", "min_group_size", "eps", "global_mean", "global_std", "rows"}

total_rows = 0
multi_step_found = False
# True once at least one single multi-step trajectory carries >= 2 distinct non-zero
# per-step oracle rewards. Diversity must be proven within one sampled episode, not
# aggregated across separate uniform episodes.
episode_reward_diversity_found = False
print(f"\nEvidence: {len(batches)} batches from {evidence_file}")

_NUMERIC_ROW_FIELDS = ("oracle_reward", "outcome_bonus", "effective_reward", "advantage")


def _finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


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

    # Finite numerics — reject NaN/inf before any arithmetic, since abs(NaN) > tol
    # is False and would silently pass forged values.
    finite_ok = True
    for fld in ("global_mean", "global_std"):
        if not _finite(batch[fld]):
            errors.append(f"Batch {bi}: non-finite {fld}={batch[fld]}")
            finite_ok = False
    for ri, row in enumerate(rows):
        for fld in _NUMERIC_ROW_FIELDS:
            if not _finite(row[fld]):
                errors.append(f"Batch {bi} row {ri}: non-finite {fld}={row[fld]}")
                finite_ok = False
        if row.get("prompt_len") is not None and not _finite(row["prompt_len"]):
            errors.append(f"Batch {bi} row {ri}: non-finite prompt_len={row['prompt_len']}")
            finite_ok = False
    if not finite_ok:
        continue  # arithmetic on this batch is unsafe

    # Normalization contract — enforce the fixed smoke parameters independently rather
    # than trusting the emitted metadata, which would otherwise let a forged
    # min_group_size silently change the normalization algorithm.
    if min_group_size != CONTRACT_MIN_GROUP_SIZE:
        errors.append(
            f"Batch {bi}: min_group_size={min_group_size} != contract {CONTRACT_MIN_GROUP_SIZE}")
    if abs(eps - CONTRACT_EPS) > 1e-20:
        errors.append(f"Batch {bi}: eps={eps} != contract {CONTRACT_EPS}")

    # Outcome bonus recomputed from terminal flags (success -> +scale, else 0). This
    # rejects both a spurious bonus on a non-terminal/failed row and a missing bonus on
    # a successful terminal row.
    expected_bonus = np.array(
        [OUTCOME_REWARD_SCALE if (r["is_terminal"] and r["terminal_success"]) else 0.0
         for r in rows], dtype=np.float64)
    for ri, (row, eb) in enumerate(zip(rows, expected_bonus)):
        if abs(row["outcome_bonus"] - eb) > 1e-4:
            if not row["is_terminal"]:
                errors.append(
                    f"Batch {bi} row {ri}: non-terminal has outcome_bonus="
                    f"{row['outcome_bonus']} (expected 0)")
            else:
                errors.append(
                    f"Batch {bi} row {ri}: terminal (success={row['terminal_success']}) "
                    f"outcome_bonus={row['outcome_bonus']} != expected {eb}")

    # Effective reward must equal oracle + recomputed bonus.
    oracle = np.array([r["oracle_reward"] for r in rows], dtype=np.float64)
    eff = oracle + expected_bonus
    for ri, (row, e_eff) in enumerate(zip(rows, eff)):
        if abs(row["effective_reward"] - e_eff) > 1e-4:
            errors.append(
                f"Batch {bi} row {ri}: effective_reward={row['effective_reward']:.6f} "
                f"!= oracle+bonus={e_eff:.6f}")

    # Batch-wide statistics recomputed from the (recomputed) effective rewards, then
    # cross-checked against emitted metadata. Recomputed values drive the fallback.
    ti = np.array([r["turn_index"] for r in rows], dtype=np.int32)
    recomputed_mean = float(eff.mean())
    recomputed_std = float(eff.std() + CONTRACT_EPS)
    if abs(recomputed_mean - global_mean) > 1e-4:
        errors.append(
            f"Batch {bi}: emitted global_mean={global_mean:.6f} != "
            f"recomputed-from-rows {recomputed_mean:.6f}")
    if abs(recomputed_std - global_std) > 1e-4:
        errors.append(
            f"Batch {bi}: emitted global_std={global_std:.6f} != "
            f"recomputed-from-rows {recomputed_std:.6f}")

    exp_adv = np.zeros(len(rows), dtype=np.float64)
    for t in np.unique(ti):
        mask = ti == t
        group = eff[mask]
        if len(group) >= CONTRACT_MIN_GROUP_SIZE:
            mean_t, std_t = group.mean(), group.std() + CONTRACT_EPS
        else:
            mean_t, std_t = recomputed_mean, recomputed_std
        exp_adv[mask] = (group - mean_t) / std_t

    for ri, (row, exp) in enumerate(zip(rows, exp_adv)):
        if abs(row["advantage"] - exp) > 1e-3:
            errors.append(
                f"Batch {bi} row {ri}: advantage={row['advantage']:.6f} != recomputed={exp:.6f}"
            )

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

        # Per-episode diversity: this single trajectory's own distinct non-zero oracle
        # rewards must reach two for it to count. Two separate uniform episodes do not.
        traj_nonzero = {round(float(r["oracle_reward"]), 6)
                        for r in by_turn if abs(r["oracle_reward"]) > 1e-9}
        if len(traj_nonzero) >= 2:
            episode_reward_diversity_found = True

        # Locality requires real captured text on every step of a multi-step trajectory.
        for r in by_turn:
            if not r.get("prompt_prefix"):
                errors.append(
                    f"Batch {bi} traj {uid}: empty prompt_prefix at turn {r['turn_index']} "
                    f"(cannot verify prompt locality)")
            if not r.get("action_prefix"):
                errors.append(
                    f"Batch {bi} traj {uid}: empty action_prefix at turn {r['turn_index']} "
                    f"(cannot verify prompt locality)")

        # Prompt locality: no EARLIER action may appear in ANY later prompt. Compare both
        # the full earlier response and its extracted final <action> tag, and exclude
        # anything already present in the trajectory's baseline (turn-0) prompt so that
        # static prompt-template action-format text is not a false positive.
        baseline_prompt = by_turn[0].get("prompt_prefix", "") or ""
        for j in range(1, len(by_turn)):
            later_prompt = by_turn[j].get("prompt_prefix", "") or ""
            for i in range(j):
                earlier_full = by_turn[i].get("action_prefix", "") or ""
                needles = []
                if len(earlier_full) >= 5:
                    needles.append(earlier_full)
                tag = _final_action_tag(earlier_full)
                if tag:
                    needles.append(tag)
                for needle in needles:
                    if needle in later_prompt and needle not in baseline_prompt:
                        errors.append(
                            f"Batch {bi} traj {uid}: action from turn "
                            f"{by_turn[i]['turn_index']} appears in prompt at turn "
                            f"{by_turn[j]['turn_index']} (history leak)")
                        break

        # Prompt bounded within trajectory (token length from masks).
        lens = [r["prompt_len"] for r in by_turn if r.get("prompt_len") is not None]
        if lens and (max(lens) - min(lens)) > 200:
            errors.append(
                f"Batch {bi} traj {uid}: prompt grew {max(lens)-min(lens)} tokens "
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

if not episode_reward_diversity_found:
    print(
        "FAIL: reward diversity: no multi-step episode carries >= 2 distinct non-zero "
        "per-step oracle rewards.", file=sys.stderr)
    print(
        "      Diversity must hold within one sampled episode; an all-equal episode "
        "(e.g. every step -1.0 from invalid parses) does not prove per-step dense "
        "rewards are preserved in training tensors.", file=sys.stderr)
    sys.exit(1)

print("PASS: multi-step trajectory confirmed")
print("PASS: per-episode per-step reward diversity confirmed")
print("PASS: reward consistency, advantage recomputation, prompt locality all verified")
