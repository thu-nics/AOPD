"""Regression tests for smoke_verify.py.

Each test passes a malformed or invalid evidence/log artifact and asserts
that smoke_verify.py exits with code 1 (not silently accepts it).
"""
import json, subprocess, sys, tempfile, os
import pytest

VERIFIER = "examples/vpr_games/smoke_verify.py"

# ── Helpers ───────────────────────────────────────────────────────────────────

_GOOD_LOG = """\
[TaskRunner] step:1 - training/global_step:1.000 - vpr/oracle_reward_mean:0.2 \
- vpr/outcome_bonus_mean:0.0 - prompt_length/mean:150 \
- critic/advantages/min:-1.0 - critic/advantages/max:1.0
[TaskRunner] step:2 - training/global_step:2.000 - vpr/oracle_reward_mean:0.2 \
- vpr/outcome_bonus_mean:0.0 - prompt_length/mean:151 \
- critic/advantages/min:-1.0 - critic/advantages/max:1.0
"""

def _good_row(traj_uid="t1", turn=0, oracle=0.5, bonus=0.0, adv=0.0, is_terminal=False,
              terminal_success=False, prompt_len=150, prompt_prefix="Board:", action_prefix="<action>5</action>"):
    return {
        "traj_uid": traj_uid,
        "turn_index": turn,
        "oracle_reward": oracle,
        "outcome_bonus": bonus,
        "effective_reward": oracle + bonus,
        "advantage": adv,
        "is_terminal": is_terminal,
        "terminal_success": terminal_success,
        "prompt_len": prompt_len,
        "prompt_prefix": prompt_prefix,
        "action_prefix": action_prefix,
    }

def _good_batch_from_rows(rows, min_group_size=4, eps=1e-8):
    import numpy as np
    eff = np.array([r["effective_reward"] for r in rows], dtype=np.float64)
    ti = np.array([r["turn_index"] for r in rows], dtype=np.int32)
    global_mean = float(eff.mean())
    global_std = float(eff.std() + eps)
    exp_adv = np.zeros(len(rows), dtype=np.float64)
    for t in np.unique(ti):
        mask = ti == t
        group = eff[mask]
        if len(group) >= min_group_size:
            mean_t, std_t = group.mean(), group.std() + eps
        else:
            mean_t, std_t = global_mean, global_std
        exp_adv[mask] = (group - mean_t) / std_t
    # Inject correct advantages
    for r, adv in zip(rows, exp_adv):
        r["advantage"] = float(adv)
    return {
        "batch_id": 0, "min_group_size": min_group_size, "eps": eps,
        "global_mean": global_mean, "global_std": global_std, "rows": rows,
    }

def _good_evidence():
    """Two batches, each with multi-step trajectories."""
    rows1 = [
        _good_row("t1", turn=0, oracle=1.0, prompt_prefix="Board: X . .", action_prefix="<action>1</action>"),
        _good_row("t1", turn=1, oracle=0.0, is_terminal=True, terminal_success=True, prompt_prefix="Board: X O ."),
        _good_row("t2", turn=0, oracle=0.0, prompt_prefix="Board: . . ."),
        _good_row("t2", turn=1, oracle=1.0, is_terminal=True, prompt_prefix="Board: . X ."),
    ]
    rows2 = [
        _good_row("t3", turn=0, oracle=-1.0, prompt_prefix="Board:"),
        _good_row("t3", turn=1, oracle=0.0, is_terminal=True, prompt_prefix="Board: ."),
    ]
    return {"batches": [_good_batch_from_rows(rows1), _good_batch_from_rows(rows2)]}


def _run(log_text=None, evidence=None, extra_args=None):
    """Run smoke_verify.py and return (returncode, stdout, stderr)."""
    with tempfile.TemporaryDirectory() as d:
        log_path = os.path.join(d, "smoke.log")
        ev_path = os.path.join(d, "ev.json")
        with open(log_path, "w") as f:
            f.write(log_text or _GOOD_LOG)
        if evidence is not None:
            with open(ev_path, "w") as f:
                json.dump(evidence, f)
            args = [sys.executable, VERIFIER, log_path, ev_path]
        else:
            args = [sys.executable, VERIFIER, log_path]
        if extra_args:
            args += extra_args
        r = subprocess.run(args, capture_output=True, text=True)
        return r.returncode, r.stdout, r.stderr


# ── Tests that must PASS ──────────────────────────────────────────────────────

class TestSmokeVerifyGoodEvidence:
    def test_good_evidence_passes(self):
        rc, out, err = _run(evidence=_good_evidence())
        assert rc == 0, f"Good evidence should pass\nstdout: {out}\nstderr: {err}"
        assert "multi-step trajectory confirmed" in out


# ── Tests that must FAIL (exit 1) ─────────────────────────────────────────────

class TestSmokeVerifyBadEvidence:

    def test_missing_evidence_arg_fails(self):
        rc, _, err = _run(evidence=None)
        assert rc == 1, "Missing evidence arg should exit 1"
        assert "mandatory" in err

    def test_empty_batches_fails(self):
        rc, _, err = _run(evidence={"batches": []})
        assert rc == 1
        assert "no 'batches'" in err or "empty" in err

    def test_missing_rows_fails(self):
        ev = {"batches": [{"batch_id": 0, "min_group_size": 4, "eps": 1e-8,
                            "global_mean": 0.0, "global_std": 1.0, "rows": []}]}
        rc, _, err = _run(evidence=ev)
        assert rc == 1, "Empty rows should exit 1"

    def test_missing_batch_fields_fails(self):
        ev = {"batches": [{"batch_id": 0, "rows": [_good_row()]}]}
        rc, _, err = _run(evidence=ev)
        assert rc == 1
        assert "missing fields" in err

    def test_missing_row_fields_fails(self):
        ev = {"batches": [{"batch_id": 0, "min_group_size": 4, "eps": 1e-8,
                            "global_mean": 0.0, "global_std": 1.0,
                            "rows": [{"traj_uid": "t1", "turn_index": 0}]}]}
        rc, _, err = _run(evidence=ev)
        assert rc == 1
        assert "missing" in err

    def test_inconsistent_effective_reward_fails(self):
        ev = _good_evidence()
        # Corrupt one row: set effective_reward to something wrong
        ev["batches"][0]["rows"][0]["effective_reward"] = 99.9
        rc, _, err = _run(evidence=ev)
        assert rc == 1
        assert "effective_reward" in err

    def test_wrong_advantage_fails(self):
        ev = _good_evidence()
        ev["batches"][0]["rows"][0]["advantage"] = 999.0
        rc, _, err = _run(evidence=ev)
        assert rc == 1
        assert "advantage" in err

    def test_non_terminal_bonus_fails(self):
        ev = _good_evidence()
        # Set non-terminal row to have non-zero bonus
        row = ev["batches"][0]["rows"][0]
        assert not row["is_terminal"]
        row["outcome_bonus"] = 1.0
        row["effective_reward"] = row["oracle_reward"] + 1.0
        # Also fix advantage to be consistent (so reward check doesn't fail first)
        # but the non-terminal bonus check should still fail
        rc, _, err = _run(evidence=ev)
        assert rc == 1
        assert "non-terminal" in err

    def test_single_turn_only_fails(self):
        rows = [_good_row("t1", turn=0), _good_row("t2", turn=0)]
        ev = {"batches": [_good_batch_from_rows(rows)]}
        rc, _, err = _run(evidence=ev)
        assert rc == 1
        assert "multi-step" in err

    def test_history_growing_prompt_fails(self):
        """Prompt length that grows by > 200 chars across turns."""
        rows = [
            _good_row("t1", turn=0, prompt_len=100),
            _good_row("t1", turn=1, prompt_len=350, is_terminal=True),  # grew 250
        ]
        ev = {"batches": [_good_batch_from_rows(rows)]}
        rc, _, err = _run(evidence=ev)
        assert rc == 1
        assert "grew" in err

    def test_history_leak_in_prompt_fails(self):
        """Prior action text verbatim in next prompt."""
        action = "<action>PLACE_MINE</action>"  # >= 10 chars
        rows = [
            _good_row("t1", turn=0, prompt_prefix="Board start", action_prefix=action),
            _good_row("t1", turn=1, prompt_prefix=f"Board {action} history", is_terminal=True),
        ]
        ev = {"batches": [_good_batch_from_rows(rows)]}
        rc, _, err = _run(evidence=ev)
        assert rc == 1
        assert "history leak" in err

    def test_malformed_json_fails(self):
        with tempfile.TemporaryDirectory() as d:
            log_path = os.path.join(d, "smoke.log")
            ev_path = os.path.join(d, "ev.json")
            with open(log_path, "w") as f:
                f.write(_GOOD_LOG)
            with open(ev_path, "w") as f:
                f.write("{not valid json}")
            r = subprocess.run([sys.executable, VERIFIER, log_path, ev_path],
                               capture_output=True, text=True)
            assert r.returncode == 1
            assert "malformed" in r.stderr

    def test_no_reward_diversity_fails(self):
        """Multi-step run where every oracle reward is the same non-zero value
        (e.g. all -1.0 from invalid parses) must fail the AC-7 diversity gate."""
        rows = [
            _good_row("t1", turn=0, oracle=-1.0, prompt_prefix="B0", action_prefix="<action>1</action>"),
            _good_row("t1", turn=1, oracle=-1.0, is_terminal=True, prompt_prefix="B1", action_prefix="<action>2</action>"),
            _good_row("t2", turn=0, oracle=-1.0, prompt_prefix="C0", action_prefix="<action>3</action>"),
            _good_row("t2", turn=1, oracle=-1.0, is_terminal=True, prompt_prefix="C1", action_prefix="<action>4</action>"),
        ]
        ev = {"batches": [_good_batch_from_rows(rows)]}
        rc, _, err = _run(evidence=ev)
        assert rc == 1
        assert "diversity" in err

    def test_missing_prompt_action_field_fails(self):
        """prompt_prefix / action_prefix are now required row fields."""
        ev = _good_evidence()
        del ev["batches"][0]["rows"][0]["prompt_prefix"]
        rc, _, err = _run(evidence=ev)
        assert rc == 1
        assert "missing" in err

    def test_empty_prompt_text_on_multistep_fails(self):
        """A multi-step trajectory with empty prompt text cannot be locality-checked."""
        ev = _good_evidence()
        ev["batches"][0]["rows"][0]["prompt_prefix"] = ""
        rc, _, err = _run(evidence=ev)
        assert rc == 1
        assert "prompt_prefix" in err

    def test_empty_action_text_on_multistep_fails(self):
        ev = _good_evidence()
        ev["batches"][0]["rows"][0]["action_prefix"] = ""
        rc, _, err = _run(evidence=ev)
        assert rc == 1
        assert "action_prefix" in err

    def test_forged_global_mean_fails(self):
        """Emitted batch-wide mean inconsistent with the rows must be rejected."""
        ev = _good_evidence()
        ev["batches"][0]["global_mean"] = 999.0
        rc, _, err = _run(evidence=ev)
        assert rc == 1
        assert "global_mean" in err

    def test_forged_global_std_fails(self):
        ev = _good_evidence()
        ev["batches"][0]["global_std"] = 999.0
        rc, _, err = _run(evidence=ev)
        assert rc == 1
        assert "global_std" in err

    def test_coordinated_forged_fallback_fails(self):
        """Fallback (group < min_group_size) with forged global stats AND advantages
        made self-consistent with those forged stats must still be rejected, because
        the verifier recomputes the stats from the rows."""
        eps = 1e-8
        rows = [
            _good_row("t1", turn=0, oracle=1.0, prompt_prefix="B0", action_prefix="<action>1</action>"),
            _good_row("t1", turn=1, oracle=-1.0, is_terminal=True, prompt_prefix="B1", action_prefix="<action>2</action>"),
        ]
        forged_mean, forged_std = 0.5, 2.0  # true mean=0.0, true std=1.0+eps
        for r in rows:
            r["advantage"] = (r["effective_reward"] - forged_mean) / forged_std
        batch = {
            "batch_id": 0, "min_group_size": 10, "eps": eps,  # force fallback for every turn
            "global_mean": forged_mean, "global_std": forged_std, "rows": rows,
        }
        rc, _, err = _run(evidence={"batches": [batch]})
        assert rc == 1
        assert "global_mean" in err or "advantage" in err
