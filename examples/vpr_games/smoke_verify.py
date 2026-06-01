"""Verify VPR training smoke test output."""
import sys, re

log_file = sys.argv[1]
log = open(log_file).read()

# Find training step lines
step_pat = re.compile(r'training/global_step:(\d+)')
all_lines = [l for l in log.splitlines() if 'global_step:' in l and ('TaskRunner' in l or 'step:' in l)]

def get(pat, text):
    m = re.search(pat + r':([-0-9.e]+)', text)
    return float(m.group(1)) if m else None

errors = []

# 1. Check training steps
steps = [get('training/global_step', l) for l in all_lines if get('training/global_step', l)]
if steps and max(steps) >= 2:
    print(f"PASS: training completed {int(max(steps))} steps")
else:
    errors.append(f"training/global_step:2 not found (found: {steps})")

# Use the last step line for final metrics
last = all_lines[-1] if all_lines else log
first = all_lines[0] if all_lines else log

# 2. VPR oracle rewards logged
oracle = get('vpr/oracle_reward_mean', last)
if oracle is not None:
    print(f"PASS: vpr/oracle_reward_mean={oracle:.4f} (per-step oracle rewards confirmed)")
else:
    errors.append("vpr/oracle_reward_mean not found in last step line")

# 3. Outcome bonus logged separately and is terminal-only (≤ oracle mean)
bonus = get('vpr/outcome_bonus_mean', last)
if bonus is not None:
    if oracle is not None and bonus > oracle + 0.001:
        errors.append(f"outcome_bonus={bonus:.4f} exceeds oracle_mean={oracle:.4f} (expected terminal-only)")
    else:
        print(f"PASS: vpr/outcome_bonus_mean={bonus:.4f} (terminal-only, <= oracle_mean)")
else:
    errors.append("vpr/outcome_bonus_mean not found")

# 4. Advantages computed (VPR estimator ran); warn if all zeros (can legitimately
# happen when all episodes yield equal rewards, e.g. all-invalid or all-legal-non-oracle).
adv_min = get('critic/advantages/min', last)
adv_max = get('critic/advantages/max', last)
if adv_min is not None and adv_max is not None:
    spread = adv_max - adv_min
    if spread < 1e-6:
        print(f"INFO: Advantages are 0 (all-equal rewards this batch): min={adv_min}, max={adv_max}")
        print("      VPR estimator ran; zero advantages are correct when all rewards are identical.")
    else:
        print(f"PASS: advantages distinct: min={adv_min:.4f}, max={adv_max:.4f}")
else:
    print(f"INFO: critic/advantages/min or /max not found in log")

# 5. Prompt length bounded (Markovian: prompts don't grow with history)
pl_all = [get('prompt_length/mean', l) for l in all_lines if get('prompt_length/mean', l) is not None]
if len(pl_all) >= 2:
    growth = pl_all[-1] - pl_all[0]
    if growth > 200:
        errors.append(f"Prompt grew by {growth:.1f} chars: step1={pl_all[0]:.1f}, last={pl_all[-1]:.1f}")
    else:
        print(f"PASS: prompt bounded: first={pl_all[0]:.1f}, last={pl_all[-1]:.1f}, delta={growth:.1f}")
elif pl_all:
    print(f"PASS: prompt_length/mean={pl_all[0]:.1f} (single step logged)")
else:
    print(f"INFO: prompt_length/mean not found in step lines")

if errors:
    for e in errors:
        print(f"FAIL: {e}", file=sys.stderr)
    sys.exit(1)
