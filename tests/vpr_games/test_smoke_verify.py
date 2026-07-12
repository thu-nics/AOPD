"""Regression tests for smoke_verify.py.

Each test passes a malformed or invalid evidence/log artifact and asserts
that smoke_verify.py exits with code 1 (not silently accepts it).
"""
import json, subprocess, sys, tempfile, os
import pytest

VERIFIER = "examples/vpr_games/smoke/smoke_verify.py"

# ── Helpers ───────────────────────────────────────────────────────────────────

_GOOD_LOG = """\
[TaskRunner] step:1 - training/global_step:1.000 - vpr/oracle_reward_mean:0.2 \
- vpr/outcome_bonus_mean:0.0 - prompt_length/mean:150 \
- critic/advantages/min:-1.0 - critic/advantages/max:1.0
[TaskRunner] step:2 - training/global_step:2.000 - vpr/oracle_reward_mean:0.2 \
- vpr/outcome_bonus_mean:0.0 - prompt_length/mean:151 \
- critic/advantages/min:-1.0 - critic/advantages/max:1.0
"""

def _good_row(traj_uid="t1", turn=0, oracle=0.0, adv=0.0, is_terminal=False,
              terminal_success=False, prompt_len=150, prompt_prefix="Board:", action_prefix="<action>5</action>"):
    # Outcome bonus is fully determined by the terminal flags (success -> +1.0), matching
    # the verifier's recomputation contract.
    bonus = 1.0 if (is_terminal and terminal_success) else 0.0
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
    """Two batches, each with multi-step trajectories. Trajectory t1 carries two
    distinct non-zero per-step oracle rewards (+1.0 then -1.0), satisfying the
    per-episode reward-diversity requirement. Turns are contiguous from 0, terminal
    rows are last, oracle rewards are in {-1,0,1}, and per-turn prompts are equal-length
    and non-substring (so they do not trip the observation-only / growth checks)."""
    rows1 = [
        _good_row("t1", turn=0, oracle=1.0, prompt_prefix="BOARD turn0 A", action_prefix="<action>1</action>"),
        _good_row("t1", turn=1, oracle=-1.0, is_terminal=True, prompt_prefix="BOARD turn1 A", action_prefix="<action>2</action>"),
        _good_row("t2", turn=0, oracle=0.0, prompt_prefix="BOARD turn0 C", action_prefix="<action>7</action>"),
        _good_row("t2", turn=1, oracle=1.0, is_terminal=True, prompt_prefix="BOARD turn1 C", action_prefix="<action>8</action>"),
    ]
    rows2 = [
        _good_row("t3", turn=0, oracle=-1.0, prompt_prefix="BOARD turn0 D", action_prefix="<action>3</action>"),
        _good_row("t3", turn=1, oracle=0.0, is_terminal=True, prompt_prefix="BOARD turn1 D", action_prefix="<action>4</action>"),
    ]
    b0 = _good_batch_from_rows(rows1)
    b1 = _good_batch_from_rows(rows2)
    b1["batch_id"] = 1  # batch_id must equal its ordered position
    return {"batches": [b0, b1]}


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
        (e.g. all -1.0 from invalid parses) must fail the reward-diversity gate."""
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


# ── Round 2 hardening: per-episode diversity, all-pairs locality, contract/finiteness ──

class TestSmokeVerifyHardening:

    def test_cross_trajectory_only_diversity_fails(self):
        """Two multi-step episodes that are each internally uniform (t1 all +1, t2 all -1)
        must fail: diversity has to hold WITHIN one sampled episode, not across episodes."""
        rows = [
            _good_row("t1", turn=0, oracle=1.0, prompt_prefix="A0", action_prefix="<action>1</action>"),
            _good_row("t1", turn=1, oracle=1.0, is_terminal=True, prompt_prefix="A1", action_prefix="<action>2</action>"),
            _good_row("t2", turn=0, oracle=-1.0, prompt_prefix="B0", action_prefix="<action>3</action>"),
            _good_row("t2", turn=1, oracle=-1.0, is_terminal=True, prompt_prefix="B1", action_prefix="<action>4</action>"),
        ]
        ev = {"batches": [_good_batch_from_rows(rows)]}
        rc, _, err = _run(evidence=ev)
        assert rc == 1
        assert "distinct non-zero" in err

    def test_non_adjacent_history_leak_fails(self):
        """A turn-0 action leaked into the turn-2 prompt (not the immediately prior turn)
        must be caught."""
        leak = "<action>reveal 1 1</action>"
        rows = [
            _good_row("t1", turn=0, oracle=1.0, prompt_prefix="Board0", action_prefix=leak),
            _good_row("t1", turn=1, oracle=-1.0, prompt_prefix="Board1 clean", action_prefix="<action>reveal 2 2</action>"),
            _good_row("t1", turn=2, oracle=1.0, is_terminal=True,
                      prompt_prefix=f"Board2 {leak} echoed", action_prefix="<action>reveal 3 3</action>"),
        ]
        ev = {"batches": [_good_batch_from_rows(rows)]}
        rc, _, err = _run(evidence=ev)
        assert rc == 1
        assert "history leak" in err

    def test_post_char50_action_tag_leak_fails(self):
        """The final <action> tag sits past character 50 of the model response; a
        prefix[:50] check would miss it. The extracted-tag leak must still be caught."""
        long_think = "<think>" + ("reasoning " * 8) + "</think>"
        action_resp = long_think + "<action>flag 4 4</action>"
        assert action_resp.index("<action>flag") > 50
        rows = [
            _good_row("t1", turn=0, oracle=1.0, prompt_prefix="Board0", action_prefix=action_resp),
            _good_row("t1", turn=1, oracle=-1.0, is_terminal=True,
                      prompt_prefix="Board1 <action>flag 4 4</action> leaked",
                      action_prefix="<action>flag 5 5</action>"),
        ]
        ev = {"batches": [_good_batch_from_rows(rows)]}
        rc, _, err = _run(evidence=ev)
        assert rc == 1
        assert "history leak" in err

    def test_forged_min_group_size_fails(self):
        ev = _good_evidence()
        ev["batches"][0]["min_group_size"] = 1
        rc, _, err = _run(evidence=ev)
        assert rc == 1
        assert "min_group_size" in err

    def test_forged_eps_fails(self):
        ev = _good_evidence()
        ev["batches"][0]["eps"] = 1e-3
        rc, _, err = _run(evidence=ev)
        assert rc == 1
        assert "eps" in err

    def test_missing_success_terminal_bonus_fails(self):
        """A successful terminal row that omits the default +1.0 bonus must be rejected,
        even though effective_reward and advantages stay internally consistent."""
        rows = [
            _good_row("t1", turn=0, oracle=1.0, prompt_prefix="A0", action_prefix="<action>1</action>"),
            _good_row("t1", turn=1, oracle=-1.0, is_terminal=True, terminal_success=True,
                      prompt_prefix="A1", action_prefix="<action>2</action>"),
        ]
        batch = _good_batch_from_rows(rows)
        srow = batch["rows"][1]
        assert srow["is_terminal"] and srow["terminal_success"]
        srow["outcome_bonus"] = 0.0  # drop the required +1.0 (effective stays oracle+1.0)
        rc, _, err = _run(evidence={"batches": [batch]})
        assert rc == 1
        assert "outcome_bonus" in err

    def test_nan_field_fails(self):
        """NaN must be rejected before arithmetic (abs(NaN) > tol is False)."""
        ev = _good_evidence()
        ev["batches"][0]["rows"][0]["advantage"] = float("nan")
        rc, _, err = _run(evidence=ev)
        assert rc == 1
        assert "finite" in err


# ── Round 3 hardening: strict schema / type validation before arithmetic ──

class TestSmokeVerifySchema:

    def test_non_object_root_fails(self):
        rc, _, err = _run(evidence=[1, 2, 3])
        assert rc == 1
        assert "root must be a JSON object" in err

    def test_batch_not_object_fails(self):
        rc, _, err = _run(evidence={"batches": [42]})
        assert rc == 1
        assert "must be a JSON object" in err

    def test_rows_not_list_fails(self):
        ev = _good_evidence()
        ev["batches"][0]["rows"] = "not-a-list"
        rc, _, err = _run(evidence=ev)
        assert rc == 1
        assert "rows" in err

    def test_eps_nan_fails(self):
        """eps=NaN must be rejected; it would otherwise slip past abs(eps-CONTRACT)>tol."""
        ev = _good_evidence()
        ev["batches"][0]["eps"] = float("nan")
        rc, _, err = _run(evidence=ev)
        assert rc == 1
        assert "eps" in err

    def test_string_terminal_flag_success_forgery_fails(self):
        """A terminal row with a STRING terminal_success ("false") plus a forged +1.0
        bonus must be rejected: a non-bool truthy flag would otherwise be read as success."""
        rows = [
            _good_row("t1", turn=0, oracle=1.0, prompt_prefix="A0", action_prefix="<action>1</action>"),
            _good_row("t1", turn=1, oracle=-1.0, is_terminal=True, prompt_prefix="A1", action_prefix="<action>2</action>"),
        ]
        batch = _good_batch_from_rows(rows)
        srow = batch["rows"][1]
        srow["terminal_success"] = "false"          # string — truthy if not type-checked
        srow["outcome_bonus"] = 1.0                  # forged success bonus
        srow["effective_reward"] = srow["oracle_reward"] + 1.0
        rc, _, err = _run(evidence={"batches": [batch]})
        assert rc == 1
        assert "terminal_success" in err

    def test_fractional_turn_index_fails(self):
        ev = _good_evidence()
        ev["batches"][0]["rows"][0]["turn_index"] = 0.1
        rc, _, err = _run(evidence=ev)
        assert rc == 1
        assert "turn_index" in err

    def test_null_prompt_len_fails(self):
        """All-null prompt_len must fail: it would otherwise bypass the prompt-growth proof."""
        ev = _good_evidence()
        for b in ev["batches"]:
            for r in b["rows"]:
                r["prompt_len"] = None
        rc, _, err = _run(evidence=ev)
        assert rc == 1
        assert "prompt_len" in err

    def test_non_integer_min_group_size_fails(self):
        ev = _good_evidence()
        ev["batches"][0]["min_group_size"] = 4.0  # float, not int
        rc, _, err = _run(evidence=ev)
        assert rc == 1
        assert "min_group_size" in err


# ── Round 4 hardening: trajectory semantics, oracle domain, observation-only prompts ──

class TestSmokeVerifyTrajectorySemantics:

    def test_duplicate_turn_index_fails(self):
        rows = [
            _good_row("t1", turn=0, oracle=1.0, prompt_prefix="P0", action_prefix="<action>1</action>"),
            _good_row("t1", turn=0, oracle=-1.0, prompt_prefix="P0b", action_prefix="<action>2</action>"),
        ]
        rc, _, err = _run(evidence={"batches": [_good_batch_from_rows(rows)]})
        assert rc == 1
        assert "duplicate turn" in err

    def test_non_contiguous_turns_fails(self):
        rows = [
            _good_row("t1", turn=0, oracle=1.0, prompt_prefix="P0", action_prefix="<action>1</action>"),
            _good_row("t1", turn=2, oracle=-1.0, is_terminal=True, prompt_prefix="P2", action_prefix="<action>2</action>"),
        ]
        rc, _, err = _run(evidence={"batches": [_good_batch_from_rows(rows)]})
        assert rc == 1
        assert "contiguous" in err

    def test_terminal_not_last_fails(self):
        rows = [
            _good_row("t1", turn=0, oracle=1.0, is_terminal=True, prompt_prefix="P0", action_prefix="<action>1</action>"),
            _good_row("t1", turn=1, oracle=-1.0, prompt_prefix="P1", action_prefix="<action>2</action>"),
        ]
        rc, _, err = _run(evidence={"batches": [_good_batch_from_rows(rows)]})
        assert rc == 1
        assert "terminal row before" in err

    def test_success_on_non_terminal_fails(self):
        rows = [
            _good_row("t1", turn=0, oracle=1.0, is_terminal=False, terminal_success=True,
                      prompt_prefix="P0", action_prefix="<action>1</action>"),
            _good_row("t1", turn=1, oracle=-1.0, is_terminal=True, prompt_prefix="P1", action_prefix="<action>2</action>"),
        ]
        rc, _, err = _run(evidence={"batches": [_good_batch_from_rows(rows)]})
        assert rc == 1
        assert "terminal_success on non-terminal" in err

    def test_oracle_reward_out_of_domain_fails(self):
        rows = [
            _good_row("t1", turn=0, oracle=2.0, prompt_prefix="P0", action_prefix="<action>1</action>"),
            _good_row("t1", turn=1, oracle=-3.0, is_terminal=True, prompt_prefix="P1", action_prefix="<action>2</action>"),
        ]
        rc, _, err = _run(evidence={"batches": [_good_batch_from_rows(rows)]})
        assert rc == 1
        assert "oracle_reward" in err

    def test_oversized_turn_index_fails_cleanly(self):
        """A gigantic integer turn_index must fail validation, not raise OverflowError."""
        ev = _good_evidence()
        ev["batches"][0]["rows"][0]["turn_index"] = 10 ** 400
        rc, _, err = _run(evidence=ev)
        assert rc == 1
        assert "turn_index" in err
        assert "Traceback" not in err and "OverflowError" not in err

    def test_prior_prompt_substring_leak_fails(self):
        """A later prompt embedding the full earlier prompt (appended observation) fails."""
        rows = [
            _good_row("t1", turn=0, oracle=1.0, prompt_prefix="CURRENT BOARD A",
                      action_prefix="<action>1</action>"),
            _good_row("t1", turn=1, oracle=-1.0, is_terminal=True,
                      prompt_prefix="CURRENT BOARD B\nPREVIOUS OBSERVATION: CURRENT BOARD A",
                      action_prefix="<action>2</action>"),
        ]
        rc, _, err = _run(evidence={"batches": [_good_batch_from_rows(rows)]})
        assert rc == 1
        assert "history leak" in err

    def test_char_length_growth_with_forged_constant_prompt_len_fails(self):
        """Char-length growth must be caught even when the emitted token prompt_len is a
        forged constant (so prompt_len alone cannot be the growth proof)."""
        rows = [
            _good_row("t1", turn=0, oracle=1.0, prompt_len=100, prompt_prefix="A" * 10,
                      action_prefix="<action>1</action>"),
            _good_row("t1", turn=1, oracle=-1.0, is_terminal=True, prompt_len=100,
                      prompt_prefix="B" * 300, action_prefix="<action>2</action>"),
        ]
        rc, _, err = _run(evidence={"batches": [_good_batch_from_rows(rows)]})
        assert rc == 1
        assert "char length grew" in err

    def test_unordered_batch_id_fails(self):
        ev = _good_evidence()
        ev["batches"][1]["batch_id"] = 0  # should equal its position (1)
        rc, _, err = _run(evidence=ev)
        assert rc == 1
        assert "batch_id" in err


class TestSmokeVerifyOversizedNumerics:
    """Oversized integers in any arithmetic field must fail validation cleanly, not raise
    an OverflowError when later converted to float/NumPy."""

    _HUGE = 10 ** 400  # far beyond float64 max; float(_HUGE) raises OverflowError

    @pytest.mark.parametrize("field", ["oracle_reward", "outcome_bonus",
                                       "effective_reward", "advantage"])
    def test_oversized_row_numeric_fails_cleanly(self, field):
        ev = _good_evidence()
        ev["batches"][0]["rows"][0][field] = self._HUGE
        rc, _, err = _run(evidence=ev)
        assert rc == 1
        assert "OverflowError" not in err and "Traceback" not in err
        assert "finite" in err

    @pytest.mark.parametrize("field", ["eps", "global_mean", "global_std"])
    def test_oversized_batch_numeric_fails_cleanly(self, field):
        ev = _good_evidence()
        ev["batches"][0][field] = self._HUGE
        rc, _, err = _run(evidence=ev)
        assert rc == 1
        assert "OverflowError" not in err and "Traceback" not in err
        assert "finite" in err


class TestSmokeVerifyLogFiniteness:
    """Layer-1 aggregate log metrics must reject present-but-non-finite values cleanly
    (a logged 1e999 -> inf must not be accepted or crash int())."""

    def test_infinite_global_step_fails_cleanly(self):
        log = _GOOD_LOG.replace("training/global_step:2.000", "training/global_step:1e999")
        rc, _, err = _run(log_text=log, evidence=_good_evidence())
        assert rc == 1
        assert "OverflowError" not in err and "Traceback" not in err
        assert "not finite" in err

    def test_infinite_oracle_reward_mean_fails(self):
        log = _GOOD_LOG.replace("vpr/oracle_reward_mean:0.2", "vpr/oracle_reward_mean:1e999")
        rc, _, err = _run(log_text=log, evidence=_good_evidence())
        assert rc == 1
        assert "not finite" in err and "Traceback" not in err

    def test_infinite_outcome_bonus_mean_fails(self):
        log = _GOOD_LOG.replace("vpr/outcome_bonus_mean:0.0", "vpr/outcome_bonus_mean:1e999")
        rc, _, err = _run(log_text=log, evidence=_good_evidence())
        assert rc == 1
        assert "not finite" in err and "Traceback" not in err

    def test_infinite_advantage_max_fails(self):
        log = _GOOD_LOG.replace("critic/advantages/max:1.0", "critic/advantages/max:1e999")
        rc, _, err = _run(log_text=log, evidence=_good_evidence())
        assert rc == 1
        assert "not finite" in err and "Traceback" not in err

    def test_infinite_prompt_length_mean_fails(self):
        log = _GOOD_LOG.replace("prompt_length/mean:151", "prompt_length/mean:1e999")
        rc, _, err = _run(log_text=log, evidence=_good_evidence())
        assert rc == 1
        assert "not finite" in err and "Traceback" not in err


class TestSmokeVerifyLogStrictTokens:
    """Layer-1 parsing must use strict whole-token parsing: textual nan/inf and malformed
    numeric suffixes on any present metric (including the optional advantage/prompt
    diagnostics) must fail cleanly, not be silently ignored or accepted via a numeric prefix."""

    # (original "key:value" substring in _GOOD_LOG, bad replacement value token)
    _CASES = [
        ("training/global_step:2.000", "nan"),
        ("training/global_step:2.000", "inf"),
        ("training/global_step:2.000", "2.000junk"),
        ("vpr/oracle_reward_mean:0.2", "nan"),
        ("vpr/oracle_reward_mean:0.2", "inf"),
        ("vpr/oracle_reward_mean:0.2", "0.2junk"),
        ("vpr/outcome_bonus_mean:0.0", "nan"),
        ("vpr/outcome_bonus_mean:0.0", "inf"),
        ("vpr/outcome_bonus_mean:0.0", "0.0junk"),
        ("critic/advantages/max:1.0", "nan"),
        ("critic/advantages/max:1.0", "inf"),
        ("critic/advantages/max:1.0", "1.0junk"),
        ("critic/advantages/min:-1.0", "nan"),
        ("critic/advantages/min:-1.0", "inf"),
        ("prompt_length/mean:151", "nan"),
        ("prompt_length/mean:151", "inf"),
        ("prompt_length/mean:151", "151junk"),
    ]

    @pytest.mark.parametrize("orig,bad", _CASES)
    def test_malformed_layer1_token_fails(self, orig, bad):
        key = orig.split(":", 1)[0]
        log = _GOOD_LOG.replace(orig, f"{key}:{bad}")
        assert log != _GOOD_LOG, "test setup: substitution did not apply"
        rc, _, err = _run(log_text=log, evidence=_good_evidence())
        assert rc == 1
        assert "OverflowError" not in err and "Traceback" not in err

    def test_malformed_earlier_global_step_with_valid_later_fails(self):
        """A non-finite earlier step must fail even when a later valid step exists."""
        log = _GOOD_LOG.replace("training/global_step:1.000", "training/global_step:nan")
        rc, _, err = _run(log_text=log, evidence=_good_evidence())
        assert rc == 1
        assert "not finite" in err and "Traceback" not in err


class TestSmokeVerifyLogOccurrence:
    """Exact-key, all-occurrence, all-line Layer-1 validation: duplicate-on-line,
    prefixed-key shadowing, malformed earlier-step diagnostics, empty tokens."""

    @staticmethod
    def _edit_first_line(orig, bad):
        """Return _GOOD_LOG with only the FIRST step line's `orig` substring replaced."""
        lines = _GOOD_LOG.rstrip("\n").split("\n")
        lines[0] = lines[0].replace(orig, bad)
        return "\n".join(lines) + "\n"

    @pytest.mark.parametrize("orig,bad", [
        ("training/global_step:2.000", "training/global_step:2.000 training/global_step:nan"),
        ("vpr/oracle_reward_mean:0.2", "vpr/oracle_reward_mean:0.2 vpr/oracle_reward_mean:0.2junk"),
        ("vpr/outcome_bonus_mean:0.0", "vpr/outcome_bonus_mean:0.0 vpr/outcome_bonus_mean:inf"),
        ("critic/advantages/max:1.0", "critic/advantages/max:1.0 critic/advantages/max:nan"),
        ("prompt_length/mean:151", "prompt_length/mean:151 prompt_length/mean:151junk"),
    ])
    def test_duplicate_metric_on_one_line_fails(self, orig, bad):
        """A valid token followed by a malformed duplicate on the same line is ambiguous."""
        log = _GOOD_LOG.replace(orig, bad)
        rc, _, err = _run(log_text=log, evidence=_good_evidence())
        assert rc == 1
        assert "Traceback" not in err and "OverflowError" not in err

    def test_prefixed_key_shadowing_fails(self):
        """A fake prefixed key must not shadow a malformed exact key."""
        log = _GOOD_LOG.replace(
            "vpr/oracle_reward_mean:0.2",
            "fakevpr/oracle_reward_mean:0.2 vpr/oracle_reward_mean:nan")
        rc, _, err = _run(log_text=log, evidence=_good_evidence())
        assert rc == 1
        assert "not finite" in err and "Traceback" not in err

    def test_earlier_step_malformed_oracle_with_valid_final_fails(self):
        log = self._edit_first_line("vpr/oracle_reward_mean:0.2", "vpr/oracle_reward_mean:nan")
        rc, _, err = _run(log_text=log, evidence=_good_evidence())
        assert rc == 1
        assert "not finite" in err and "Traceback" not in err

    def test_earlier_step_malformed_outcome_with_valid_final_fails(self):
        log = self._edit_first_line("vpr/outcome_bonus_mean:0.0", "vpr/outcome_bonus_mean:0.0junk")
        rc, _, err = _run(log_text=log, evidence=_good_evidence())
        assert rc == 1
        assert "not finite" in err and "Traceback" not in err

    def test_earlier_step_malformed_advantage_with_valid_final_fails(self):
        log = self._edit_first_line("critic/advantages/max:1.0", "critic/advantages/max:nan")
        rc, _, err = _run(log_text=log, evidence=_good_evidence())
        assert rc == 1
        assert "not finite" in err and "Traceback" not in err

    def test_empty_metric_token_fails(self):
        """`critic/advantages/max:` with no value token (end of line) is present-but-empty."""
        log = _GOOD_LOG.replace("critic/advantages/max:1.0", "critic/advantages/max:")
        rc, _, err = _run(log_text=log, evidence=_good_evidence())
        assert rc == 1
        assert "empty value token" in err and "Traceback" not in err

    def test_advantages_min_suffix_junk_fails(self):
        log = _GOOD_LOG.replace("critic/advantages/min:-1.0", "critic/advantages/min:-1.0junk")
        rc, _, err = _run(log_text=log, evidence=_good_evidence())
        assert rc == 1
        assert "not finite" in err and "Traceback" not in err


class TestSmokeVerifyLogShapeAndDomain:
    """Shape-based trainer-record selection and semantic-domain validation for Layer-1
    aggregate metrics."""

    @staticmethod
    def _append(line):
        return _GOOD_LOG.rstrip("\n") + "\n" + line + "\n"

    @staticmethod
    def _edit_first_line(orig, bad):
        lines = _GOOD_LOG.rstrip("\n").split("\n")
        lines[0] = lines[0].replace(orig, bad)
        return "\n".join(lines) + "\n"

    def test_trainer_line_without_global_step_fails(self):
        """A TaskRunner step record with no exact global-step token must fail (it is now
        selected by record shape, not by a global_step substring)."""
        bad = ("[TaskRunner] step:3 - vpr/oracle_reward_mean:0.2 - "
               "vpr/outcome_bonus_mean:0.0 - prompt_length/mean:150")
        rc, _, err = _run(log_text=self._append(bad), evidence=_good_evidence())
        assert rc == 1
        assert "no valid" in err and "Traceback" not in err

    def test_last_line_prefixed_fake_global_step_fails(self):
        bad = ("[TaskRunner] step:3 - fake/training/global_step:3.000 - "
               "vpr/oracle_reward_mean:0.2 - vpr/outcome_bonus_mean:0.0")
        rc, _, err = _run(log_text=self._append(bad), evidence=_good_evidence())
        assert rc == 1
        assert "no valid" in err and "Traceback" not in err

    def test_fractional_global_step_fails(self):
        log = _GOOD_LOG.replace("training/global_step:2.000", "training/global_step:2.5")
        rc, _, err = _run(log_text=log, evidence=_good_evidence())
        assert rc == 1
        assert "nonnegative integer" in err and "Traceback" not in err

    def test_negative_global_step_fails(self):
        log = _GOOD_LOG.replace("training/global_step:2.000", "training/global_step:-1")
        rc, _, err = _run(log_text=log, evidence=_good_evidence())
        assert rc == 1
        assert "nonnegative integer" in err and "Traceback" not in err

    def test_oracle_mean_out_of_range_fails(self):
        log = _GOOD_LOG.replace("vpr/oracle_reward_mean:0.2", "vpr/oracle_reward_mean:2.0")
        rc, _, err = _run(log_text=log, evidence=_good_evidence())
        assert rc == 1
        assert "oracle_reward_mean" in err and "Traceback" not in err

    def test_outcome_mean_above_one_fails(self):
        log = _GOOD_LOG.replace("vpr/outcome_bonus_mean:0.0", "vpr/outcome_bonus_mean:2.0")
        rc, _, err = _run(log_text=log, evidence=_good_evidence())
        assert rc == 1
        assert "outcome_bonus_mean" in err and "Traceback" not in err

    def test_earlier_outcome_below_zero_with_valid_final_fails(self):
        log = self._edit_first_line("vpr/outcome_bonus_mean:0.0", "vpr/outcome_bonus_mean:-1.0")
        rc, _, err = _run(log_text=log, evidence=_good_evidence())
        assert rc == 1
        assert "outcome_bonus_mean" in err and "Traceback" not in err

    def test_negative_prompt_mean_fails(self):
        log = _GOOD_LOG.replace("prompt_length/mean:151", "prompt_length/mean:-100")
        rc, _, err = _run(log_text=log, evidence=_good_evidence())
        assert rc == 1
        assert "prompt_length/mean" in err and "Traceback" not in err

    def test_reversed_advantage_bounds_fails(self):
        log = _GOOD_LOG.replace("critic/advantages/min:-1.0", "critic/advantages/min:2.0")
        rc, _, err = _run(log_text=log, evidence=_good_evidence())
        assert rc == 1
        assert "min" in err and "max" in err and "Traceback" not in err
