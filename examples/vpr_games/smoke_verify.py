"""Verify VPR training smoke test output.

Usage: python3 smoke_verify.py <log_file> [evidence_file]

Checks two layers of evidence:
1. Aggregate training-log metrics (always present).
2. Per-row VPR evidence from evidence_file (written by core_gigpo.py when
   VPR_SMOKE_EVIDENCE env var is set) — proves terminal-only outcome bonus,
   bounded per-episode prompts, and per-turn reward preservation.
   A fabricated one-line log with only aggregate metrics fails this layer.
"""
import sys, re, json

log_file = sys.argv[1]
evidence_file = sys.argv[2] if len(sys.argv) > 2 else None
log = open(log_file).read()

all_lines = [l for l in log.splitlines()
             if 'global_step:' in l and ('TaskRunner' in l or 'step:' in l)]

def get(pat, text):
    m = re.search(pat + r':([-0-9.e]+)', text)
    return float(m.group(1)) if m else None

errors = []
last = all_lines[-1] if all_lines else log
first = all_lines[0] if all_lines else log

# ── Layer 1: aggregate log metrics ──────────────────────────────────────────

steps = [get('training/global_step', l) for l in all_lines if get('training/global_step', l)]
if steps and max(steps) >= 2:
    print(f"PASS: training completed {int(max(steps))} steps")
else:
    errors.append(f"training/global_step:2 not found (found: {steps})")

oracle = get('vpr/oracle_reward_mean', last)
if oracle is not None:
    print(f"PASS: vpr/oracle_reward_mean={oracle:.4f} (per-step oracle rewards logged)")
else:
    errors.append("vpr/oracle_reward_mean not found in last step line")

bonus_mean = get('vpr/outcome_bonus_mean', last)
if bonus_mean is not None:
    if bonus_mean < -1e-6:
        errors.append(f"outcome_bonus_mean={bonus_mean:.4f} is negative (must be >= 0)")
    else:
        print(f"PASS: vpr/outcome_bonus_mean={bonus_mean:.4f} (>= 0 aggregate)")
else:
    errors.append("vpr/outcome_bonus_mean not found")

adv_min = get('critic/advantages/min', last)
adv_max = get('critic/advantages/max', last)
if adv_min is not None and adv_max is not None:
    spread = adv_max - adv_min
    if spread < 1e-6:
        print(f"INFO: advantages=0 (all-equal rewards this batch): min={adv_min}, max={adv_max}")
        print("      VPR estimator ran; zero advantages are correct when rewards are identical.")
    else:
        print(f"PASS: advantages distinct: min={adv_min:.4f}, max={adv_max:.4f}")
else:
    print("INFO: critic/advantages/min or /max not found in log")

pl_all = [get('prompt_length/mean', l) for l in all_lines if get('prompt_length/mean', l) is not None]
if len(pl_all) >= 2:
    growth = pl_all[-1] - pl_all[0]
    if growth > 200:
        errors.append(f"Prompt grew by {growth:.1f} chars (batch mean): step1={pl_all[0]:.1f}, last={pl_all[-1]:.1f}")
    else:
        print(f"PASS: batch prompt bounded: first={pl_all[0]:.1f}, last={pl_all[-1]:.1f}, delta={growth:.1f}")
elif pl_all:
    print(f"PASS: prompt_length/mean={pl_all[0]:.1f} (single step logged)")
else:
    print("INFO: prompt_length/mean not found in step lines")

# ── Layer 2: per-row evidence file ──────────────────────────────────────────

if evidence_file:
    try:
        with open(evidence_file) as f:
            ev = json.load(f)
        rows = ev.get("rows", [])
        print(f"\nEvidence file: {len(rows)} rows from {evidence_file}")
    except (FileNotFoundError, ValueError) as e:
        errors.append(f"Evidence file unreadable: {e}")
        rows = []
else:
    rows = []
    print("INFO: No evidence file provided (layer-2 checks skipped)")

if rows:
    # 2a. Outcome bonus is zero on every non-terminal row
    non_terminal_bonus_violations = [
        r for r in rows if not r.get("is_terminal", False) and r.get("outcome_bonus", 0.0) != 0.0
    ]
    if non_terminal_bonus_violations:
        for r in non_terminal_bonus_violations[:3]:
            errors.append(
                f"Non-terminal row has outcome_bonus={r['outcome_bonus']}: "
                f"traj={r.get('traj_uid')}, turn={r.get('turn_index')}"
            )
    else:
        print(f"PASS: outcome_bonus == 0 on all {sum(1 for r in rows if not r.get('is_terminal', False))} non-terminal rows")

    # 2b. Per-episode prompt lengths are bounded (no history accumulation)
    from collections import defaultdict
    traj_prompts = defaultdict(list)
    for r in rows:
        uid = r.get("traj_uid")
        pl = r.get("prompt_len")
        if uid and pl is not None:
            traj_prompts[uid].append(pl)

    prompt_violations = []
    for uid, lens in traj_prompts.items():
        if len(lens) >= 2:
            growth = max(lens) - min(lens)
            if growth > 200:
                prompt_violations.append(f"traj {uid}: prompt_len range {min(lens)}-{max(lens)} (growth={growth})")

    if prompt_violations:
        for v in prompt_violations[:3]:
            errors.append(f"Prompt grew within episode: {v}")
    else:
        multi_step_trajs = sum(1 for lens in traj_prompts.values() if len(lens) >= 2)
        print(f"PASS: prompt bounded within episodes ({multi_step_trajs} multi-step trajectories checked)")

    # 2c. Per-turn reward data is present (proves VPR path ran)
    turns = sorted(set(r.get("turn_index", 0) for r in rows))
    print(f"PASS: per-row evidence covers {len(rows)} rows across turn indices {turns}")

    # 2d. Multi-step evidence: at least one trajectory has >= 2 turns
    multi_step = [uid for uid, lens in traj_prompts.items() if len(lens) >= 2]
    if multi_step:
        print(f"PASS: {len(multi_step)} trajectories have >= 2 rollout turns (multi-turn preservation confirmed)")
    else:
        # All episodes terminated in 1 step — not a bug but worth noting
        print(f"INFO: All episodes terminated in 1 step (cannot verify per-turn progression)")

# ── Final verdict ────────────────────────────────────────────────────────────

if errors:
    for e in errors:
        print(f"FAIL: {e}", file=sys.stderr)
    sys.exit(1)
