"""
Visual test scenarios for lowlvl.py

Runs cumulative sequences of operations and produces a plot after each step,
showing the warping path (time_from → time_to) flanked by src and tgt piano rolls.
Loads via partitura with pedal_threshold=127 to match the caller pipeline.

Usage:
    pytest test_visual_lowlvl.py -v -s
    # plots saved to ./plots/
"""

import numpy as np
import copy
import pytest
import partitura as pt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

from piano_synmist.lowlvl import lowlvl, regular_na_fields
from plot_lowlvl import plot_lowlvl_state, plot_cumulative_operations

MIDI_PATH = os.path.join(os.path.dirname(__file__), "kv279_1.mid")
PLOT_DIR = os.path.join(os.path.dirname(__file__), "plots")


@pytest.fixture(scope="session")
def src_na():
    perf = pt.load_performance(MIDI_PATH, pedal_threshold=127)
    na = perf.note_array()
    na.sort(order="onset_sec")
    return na


def _fresh_ll(src_na):
    return lowlvl(copy.deepcopy(src_na), mode="runthrough")


def _pick(ll, index):
    n = ll.src_na[index]
    return float(n["onset_sec"]), int(n["pitch"]), float(n["duration_sec"])


# ===================================================================
# Scenario 1: Insertions — accumulate mistouches
# ===================================================================

class TestScenarioInsertions:
    def test_cumulative_inserts(self, src_na):
        ll = _fresh_ll(src_na)
        t0, p0, _ = _pick(ll, 60)
        t1, p1, _ = _pick(ll, 90)
        t2, p2, _ = _pick(ll, 120)

        ops = [
            {"name": f"Insert pitch 100 at t={t0:.2f}",
             "func": "pitch_insert",
             "args": dict(src_time=t0, pitch=100, duration=0.2,
                          velocity=80, midlvl_label="mistouch")},
            {"name": f"Insert pitch 102 at t={t1:.2f}",
             "func": "pitch_insert",
             "args": dict(src_time=t1, pitch=102, duration=0.15,
                          velocity=70, midlvl_label="mistouch")},
            {"name": f"Insert pitch 55 at t={t2:.2f}",
             "func": "pitch_insert",
             "args": dict(src_time=t2, pitch=55, duration=0.3,
                          velocity=90, midlvl_label="wrong_pred")},
        ]

        figs = plot_cumulative_operations(
            ll, ops,
            time_interval=(t0 - 1, t2 + 2),
            output_dir=os.path.join(PLOT_DIR, "insertions"),
            prefix="ins")

        assert len(figs) == 4  # initial + 3 steps
        assert 100 in ll.tgt_na["pitch"]
        assert 102 in ll.tgt_na["pitch"]
        assert 55 in ll.tgt_na["pitch"]


# ===================================================================
# Scenario 2: Deletions
# ===================================================================

class TestScenarioDeletions:
    def test_cumulative_deletes(self, src_na):
        ll = _fresh_ll(src_na)
        t0, p0, _ = _pick(ll, 70)
        t1, p1, _ = _pick(ll, 100)

        ops = [
            {"name": f"Delete pitch {p0} at t={t0:.2f}",
             "func": "pitch_delete",
             "args": dict(src_time=t0, pitch=p0, midlvl_label="mistouch")},
            {"name": f"Delete pitch {p1} at t={t1:.2f}",
             "func": "pitch_delete",
             "args": dict(src_time=t1, pitch=p1, midlvl_label="mistouch")},
        ]

        original_len = len(ll.tgt_na)
        figs = plot_cumulative_operations(
            ll, ops,
            time_interval=(t0 - 1, t1 + 2),
            output_dir=os.path.join(PLOT_DIR, "deletions"),
            prefix="del")

        assert len(figs) == 3
        assert len(ll.tgt_na) == original_len - 2


# ===================================================================
# Scenario 3: Time offsets (drag / silence insertion)
# ===================================================================

class TestScenarioTimeOffset:
    def test_cumulative_offsets(self, src_na):
        ll = _fresh_ll(src_na)
        t0, _, _ = _pick(ll, 80)
        t1, _, _ = _pick(ll, 160)

        ops = [
            {"name": f"Add 0.5s silence at t={t0:.2f}",
             "func": "time_offset",
             "args": dict(src_time=t0, offset_time=0.5, midlvl_label="drag")},
            {"name": f"Add 1.0s silence at t={t1:.2f}",
             "func": "time_offset",
             "args": dict(src_time=t1, offset_time=1.0, midlvl_label="drag")},
        ]

        figs = plot_cumulative_operations(
            ll, ops,
            time_interval=(t0 - 2, t1 + 4),
            output_dir=os.path.join(PLOT_DIR, "offsets"),
            prefix="off")

        assert len(figs) == 3
        # Warping path should now diverge from identity
        idx = np.fabs(ll.time_from - t1 - 1).argmin()
        assert ll.time_to[idx] > ll.time_from[idx]


# ===================================================================
# Scenario 4: Duration changes (change_note_offset)
# ===================================================================

class TestScenarioDurationChange:
    def test_cumulative_duration_changes(self, src_na):
        ll = _fresh_ll(src_na)
        t0, p0, d0 = _pick(ll, 80)
        t1, p1, d1 = _pick(ll, 130)

        ops = [
            {"name": f"Extend pitch {p0} by +0.3s at t={t0:.2f}",
             "func": "change_note_offset",
             "args": dict(src_time=t0, pitch=p0, offset_shift=0.3,
                          midlvl_label="drag")},
            {"name": f"Shorten pitch {p1} by -0.03s at t={t1:.2f}",
             "func": "change_note_offset",
             "args": dict(src_time=t1, pitch=p1, offset_shift=-0.03,
                          midlvl_label="drag")},
        ]

        figs = plot_cumulative_operations(
            ll, ops,
            time_interval=(t0 - 1, t1 + 2),
            output_dir=os.path.join(PLOT_DIR, "duration"),
            prefix="dur")

        assert len(figs) == 3


# ===================================================================
# Scenario 5: Go back (rollback) — the big one
# ===================================================================

class TestScenarioGoBack:
    def test_goback_with_plots(self, src_na):
        ll = _fresh_ll(src_na)
        t_to = float(ll.src_na["onset_sec"][100])
        t_from = float(ll.src_na["onset_sec"][180])

        # Plot initial state
        fig0 = plot_lowlvl_state(
            ll, title="Go-back: Initial state",
            time_interval=(t_to - 2, t_from + 5),
            save_path=os.path.join(PLOT_DIR, "goback", "gb_00_initial.png"))

        # Perform rollback
        notes = ll.get_notes_between(t_to, t_from)
        for n in notes:
            n["onset_sec"] -= notes[0]["onset_sec"]

        ll.go_back(src_time_to=t_to, src_time_from=t_from,
                   notes=notes, midlvl_label="rollback")

        fig1 = plot_lowlvl_state(
            ll, title=f"Go-back: rolled back from t={t_from:.1f} to t={t_to:.1f}",
            time_interval=(t_to - 2, t_from + 10),
            save_path=os.path.join(PLOT_DIR, "goback", "gb_01_after_rollback.png"))

        assert len(ll.repeat_tracker) == 1
        assert len(ll.tgt_na) > len(ll.src_na)

        plt.close('all')


# ===================================================================
# Scenario 6: Mixed operations — the full story
# ===================================================================

class TestScenarioMixed:
    def test_mixed_cumulative(self, src_na):
        """
        Realistic sequence: insert a mistouch, add a hesitation (offset),
        delete a wrong note, adjust duration, then roll back.
        """
        ll = _fresh_ll(src_na)

        t_ins, _, _ = _pick(ll, 60)
        t_off, _, _ = _pick(ll, 90)
        t_del, p_del, _ = _pick(ll, 110)
        t_dur, p_dur, _ = _pick(ll, 130)
        t_back_to = float(ll.src_na["onset_sec"][80])
        t_back_from = float(ll.src_na["onset_sec"][140])

        outdir = os.path.join(PLOT_DIR, "mixed")
        os.makedirs(outdir, exist_ok=True)
        step = 0

        def _plot(title):
            nonlocal step
            plot_lowlvl_state(ll, title=title,
                              time_interval=(t_ins - 2, t_back_from + 8),
                              save_path=os.path.join(outdir, f"mix_{step:02d}.png"))
            step += 1

        _plot("Step 0: Initial")

        # 1. Insert mistouch
        ll.pitch_insert(src_time=t_ins, pitch=100, duration=0.15,
                        velocity=80, midlvl_label="mistouch")
        _plot(f"Step 1: Insert pitch 100 at t={t_ins:.2f}")

        # 2. Hesitation (time offset)
        ll.time_offset(src_time=t_off, offset_time=0.6, midlvl_label="drag")
        _plot(f"Step 2: +0.6s offset at t={t_off:.2f}")

        # 3. Delete a note
        ll.pitch_delete(src_time=t_del, pitch=p_del, midlvl_label="mistouch")
        _plot(f"Step 3: Delete pitch {p_del} at t={t_del:.2f}")

        # 4. Adjust duration
        ll.change_note_offset(src_time=t_dur, pitch=p_dur,
                              offset_shift=0.25, midlvl_label="drag")
        _plot(f"Step 4: Extend pitch {p_dur} by +0.25s")

        # 5. Rollback
        notes = ll.get_notes_between(t_back_to, t_back_from)
        for n in notes:
            n["onset_sec"] -= notes[0]["onset_sec"]
        ll.go_back(src_time_to=t_back_to, src_time_from=t_back_from,
                    midlvl_label="rollback")
        _plot(f"Step 5: Roll back from t={t_back_from:.1f} to t={t_back_to:.1f}")

        # Assertions on final state
        assert len(ll.repeat_tracker) == 1
        assert 100 in ll.tgt_na["pitch"]
        assert any(ll.label_na["lowlvl_label"] == "pitch_insert")
        assert any(ll.label_na["lowlvl_label"] == "time_shift")
        assert any(ll.label_na["lowlvl_label"] == "pitch_delete")
        assert any(ll.label_na["lowlvl_label"] == "change_offset")

        plt.close('all')


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
