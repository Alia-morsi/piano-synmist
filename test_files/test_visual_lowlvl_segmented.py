"""
Visual test scenarios for lowlvl.py segmented-practice mode.

Runs cumulative sequences of operations on segmented practice instances
and produces a plot after each step, showing the warping path (time_from
→ time_to) flanked by src and tgt piano rolls.

Usage:
    pytest test_visual_lowlvl_segmented.py -v -s
    # plots saved to ./plots/segmented_*
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

# Segment boundaries (note indices)
SEG_A = (50, 100)
SEG_B = (150, 200)
SEG_C = (250, 300)
TIME_GAP = 3.0


@pytest.fixture(scope="session")
def src_na():
    perf = pt.load_performance(MIDI_PATH, pedal_threshold=127)
    na = perf.note_array()
    na.sort(order="onset_sec")
    return na


@pytest.fixture(scope="session")
def segments(src_na):
    return [
        (float(src_na["onset_sec"][a]), float(src_na["onset_sec"][b]))
        for a, b in [SEG_A, SEG_B, SEG_C]
    ]


def _fresh_seg(src_na, segments):
    ll = lowlvl(copy.deepcopy(src_na), mode="segmented")
    ll._create_segmented_practice(segments, time_gap=TIME_GAP)
    return ll


def _pick(ll, index):
    n = ll.src_na[index]
    return float(n["onset_sec"]), int(n["pitch"]), float(n["duration_sec"])


# ===================================================================
# Scenario 1: Initial state — all 3 segments laid out
# ===================================================================

class TestScenarioSegmentedInit:
    def test_plot_initial_state(self, src_na, segments):
        ll = _fresh_seg(src_na, segments)
        outdir = os.path.join(PLOT_DIR, "segmented_init")
        os.makedirs(outdir, exist_ok=True)

        # Full view covering all segments in src time
        fig = plot_lowlvl_state(
            ll, title="Segmented: Initial state (3 segments)",
            time_interval=(segments[0][0] - 1, segments[-1][1] + 2),
            save_path=os.path.join(outdir, "seg_init_full.png"))

        assert fig is not None

        # Zoomed view on segment A only
        fig_a = plot_lowlvl_state(
            ll, title="Segmented: Segment A detail",
            time_interval=(segments[0][0] - 0.5, segments[0][1] + 0.5),
            save_path=os.path.join(outdir, "seg_init_a.png"))

        assert fig_a is not None
        plt.close('all')


# ===================================================================
# Scenario 2: Insertions on different segments
# ===================================================================

class TestScenarioSegmentedInsertions:
    def test_cumulative_inserts(self, src_na, segments):
        ll = _fresh_seg(src_na, segments)
        t_a, _, _ = _pick(ll, SEG_A[0] + 10)
        t_b, _, _ = _pick(ll, SEG_B[0] + 10)
        t_c, _, _ = _pick(ll, SEG_C[0] + 10)

        ops = [
            {"name": f"Insert pitch 100 in seg A at t={t_a:.2f}",
             "func": "pitch_insert",
             "args": dict(src_time=t_a, pitch=100, duration=0.2,
                          velocity=80, midlvl_label="mistouch")},
            {"name": f"Insert pitch 102 in seg B at t={t_b:.2f}",
             "func": "pitch_insert",
             "args": dict(src_time=t_b, pitch=102, duration=0.15,
                          velocity=70, midlvl_label="mistouch")},
            {"name": f"Insert pitch 55 in seg C at t={t_c:.2f}",
             "func": "pitch_insert",
             "args": dict(src_time=t_c, pitch=55, duration=0.3,
                          velocity=90, midlvl_label="wrong_pred")},
        ]

        figs = plot_cumulative_operations(
            ll, ops,
            time_interval=(segments[0][0] - 1, segments[-1][1] + 2),
            output_dir=os.path.join(PLOT_DIR, "segmented_insertions"),
            prefix="seg_ins")

        assert len(figs) == 4
        assert 100 in ll.tgt_na["pitch"]
        assert 102 in ll.tgt_na["pitch"]
        assert 55 in ll.tgt_na["pitch"]
        plt.close('all')


# ===================================================================
# Scenario 3: Deletions + offsets on a single segment
# ===================================================================

class TestScenarioSegmentedDeleteAndOffset:
    def test_delete_then_offset(self, src_na, segments):
        ll = _fresh_seg(src_na, segments)
        t_del, p_del, _ = _pick(ll, SEG_A[0] + 15)
        t_off, _, _ = _pick(ll, SEG_A[0] + 25)

        ops = [
            {"name": f"Delete pitch {p_del} at t={t_del:.2f} in seg A",
             "func": "pitch_delete",
             "args": dict(src_time=t_del, pitch=p_del, midlvl_label="mistouch")},
            {"name": f"+0.7s offset at t={t_off:.2f} in seg A",
             "func": "time_offset",
             "args": dict(src_time=t_off, offset_time=0.7, midlvl_label="drag")},
        ]

        figs = plot_cumulative_operations(
            ll, ops,
            time_interval=(segments[0][0] - 1, segments[0][1] + 3),
            output_dir=os.path.join(PLOT_DIR, "segmented_del_offset"),
            prefix="seg_do")

        assert len(figs) == 3
        plt.close('all')


# ===================================================================
# Scenario 4: Go-back within segment A
# ===================================================================

class TestScenarioSegmentedGoBack:
    def test_goback_with_plots(self, src_na, segments):
        ll = _fresh_seg(src_na, segments)
        outdir = os.path.join(PLOT_DIR, "segmented_goback")
        os.makedirs(outdir, exist_ok=True)

        t_to = float(ll.src_na["onset_sec"][SEG_A[0]])
        t_from = float(ll.src_na["onset_sec"][SEG_A[1]])

        # Initial
        fig0 = plot_lowlvl_state(
            ll, title="Segmented go-back: Initial",
            time_interval=(t_to - 1, t_from + 5),
            save_path=os.path.join(outdir, "sgb_00_initial.png"))

        # Go back
        ll.go_back(src_time_to=t_to, src_time_from=t_from,
                   midlvl_label="rollback", repeat_index=0)

        fig1 = plot_lowlvl_state(
            ll, title=f"Segmented go-back: rolled back in seg A",
            time_interval=(t_to - 1, t_from + 10),
            save_path=os.path.join(outdir, "sgb_01_after_rollback.png"))

        assert len(ll.repeat_tracker) == 1
        assert len(ll.tgt_na) > len(ll.get_notes_between(t_to, t_from))
        plt.close('all')


# ===================================================================
# Scenario 5: Operations on old pass (repeat_index=1) after go-back
# ===================================================================

class TestScenarioOperationsOnRepeat:
    def test_ops_on_old_pass_with_plots(self, src_na, segments):
        ll = _fresh_seg(src_na, segments)
        outdir = os.path.join(PLOT_DIR, "segmented_repeat_ops")
        os.makedirs(outdir, exist_ok=True)
        step = 0

        def _plot(title):
            nonlocal step
            plot_lowlvl_state(ll, title=title,
                              time_interval=(segments[0][0] - 1, segments[0][1] + 15),
                              save_path=os.path.join(outdir, f"sro_{step:02d}.png"))
            step += 1

        _plot("Step 0: Initial segmented state")

        # 1. Go back in seg A
        t_to = float(ll.src_na["onset_sec"][SEG_A[0]])
        t_from = float(ll.src_na["onset_sec"][SEG_A[1]])
        ll.go_back(src_time_to=t_to, src_time_from=t_from,
                   midlvl_label="rollback", repeat_index=0)
        _plot("Step 1: Go-back in seg A")

        # 2. Insert on old pass (repeat_index=1)
        t_ins, _, _ = _pick(ll, SEG_A[0] + 10)
        ll.pitch_insert(src_time=t_ins, pitch=110, duration=0.2,
                        velocity=80, midlvl_label="mistouch", repeat_index=1)
        _plot("Step 2: Insert pitch 110 on OLD pass (repeat_index=1)")

        # 3. Insert on current pass (repeat_index=0)
        ll.pitch_insert(src_time=t_ins, pitch=112, duration=0.2,
                        velocity=80, midlvl_label="mistouch", repeat_index=0)
        _plot("Step 3: Insert pitch 112 on CURRENT pass (repeat_index=0)")

        # 4. Time offset on old pass
        t_off, _, _ = _pick(ll, SEG_A[0] + 5)
        ll.time_offset(src_time=t_off, offset_time=0.5,
                       midlvl_label="drag", repeat_index=1)
        _plot("Step 4: +0.5s offset on OLD pass")

        assert 110 in ll.tgt_na["pitch"]
        assert 112 in ll.tgt_na["pitch"]
        plt.close('all')


# ===================================================================
# Scenario 6: Multiple go-backs in same segment
# ===================================================================

class TestScenarioMultipleGobacks:
    def test_double_goback_with_plots(self, src_na, segments):
        ll = _fresh_seg(src_na, segments)
        outdir = os.path.join(PLOT_DIR, "segmented_multi_goback")
        os.makedirs(outdir, exist_ok=True)
        step = 0

        def _plot(title):
            nonlocal step
            plot_lowlvl_state(ll, title=title,
                              time_interval=(segments[0][0] - 1, segments[0][1] + 20),
                              save_path=os.path.join(outdir, f"smg_{step:02d}.png"))
            step += 1

        t_to = float(ll.src_na["onset_sec"][SEG_A[0]])
        t_from = float(ll.src_na["onset_sec"][SEG_A[1]])

        _plot("Step 0: Initial")

        # First go-back
        ll.go_back(src_time_to=t_to, src_time_from=t_from,
                   midlvl_label="rollback", repeat_index=0)
        _plot("Step 1: First go-back in seg A")

        # Insert on first pass (now in repeat_tracker)
        t_ins, _, _ = _pick(ll, SEG_A[0] + 8)
        ll.pitch_insert(src_time=t_ins, pitch=105, duration=0.2,
                        velocity=80, midlvl_label="mistouch", repeat_index=1)
        _plot("Step 2: Insert on 1st pass (repeat_index=1)")

        # Second go-back
        ll.go_back(src_time_to=t_to, src_time_from=t_from,
                   midlvl_label="rollback", repeat_index=0)
        _plot("Step 3: Second go-back in seg A")

        assert len(ll.repeat_tracker) == 2

        # Insert on 2nd repeat (the pass that just got pushed)
        ll.pitch_insert(src_time=t_ins, pitch=107, duration=0.2,
                        velocity=80, midlvl_label="mistouch", repeat_index=2)
        _plot("Step 4: Insert on 2nd pass (repeat_index=2)")

        assert 105 in ll.tgt_na["pitch"]
        assert 107 in ll.tgt_na["pitch"]
        plt.close('all')


# ===================================================================
# Scenario 7: Mixed operations across all segments
# ===================================================================

class TestScenarioSegmentedMixed:
    def test_full_mixed_scenario(self, src_na, segments):
        """
        Realistic sequence across segments:
          1. Insert mistouch in seg A
          2. Add hesitation in seg B
          3. Delete from seg B
          4. Change duration in seg C
          5. Go back in seg A
          6. Insert on old pass of seg A
          7. Go back in seg B
        """
        ll = _fresh_seg(src_na, segments)
        outdir = os.path.join(PLOT_DIR, "segmented_mixed")
        os.makedirs(outdir, exist_ok=True)
        step = 0

        def _plot(title):
            nonlocal step
            plot_lowlvl_state(ll, title=title,
                              time_interval=(segments[0][0] - 2, segments[-1][1] + 10),
                              save_path=os.path.join(outdir, f"smx_{step:02d}.png"))
            step += 1

        _plot("Step 0: Initial (3 segments)")

        # 1. Insert mistouch in seg A
        t_ins, _, _ = _pick(ll, SEG_A[0] + 8)
        ll.pitch_insert(src_time=t_ins, pitch=100, duration=0.15,
                        velocity=80, midlvl_label="mistouch")
        _plot(f"Step 1: Insert pitch 100 in seg A")

        # 2. Hesitation in seg B
        t_off, _, _ = _pick(ll, SEG_B[0] + 10)
        ll.time_offset(src_time=t_off, offset_time=0.6, midlvl_label="drag")
        _plot(f"Step 2: +0.6s hesitation in seg B")

        # 3. Delete from seg B
        t_del, p_del, _ = _pick(ll, SEG_B[0] + 20)
        ll.pitch_delete(src_time=t_del, pitch=p_del, midlvl_label="mistouch")
        _plot(f"Step 3: Delete pitch {p_del} from seg B")

        # 4. Change duration in seg C
        t_dur, p_dur, _ = _pick(ll, SEG_C[0] + 15)
        ll.change_note_offset(src_time=t_dur, pitch=p_dur,
                              offset_shift=0.25, midlvl_label="drag")
        _plot(f"Step 4: Extend pitch {p_dur} by +0.25s in seg C")

        # 5. Go back in seg A
        t_back_to = float(ll.src_na["onset_sec"][SEG_A[0]])
        t_back_from = float(ll.src_na["onset_sec"][SEG_A[1]])
        ll.go_back(src_time_to=t_back_to, src_time_from=t_back_from,
                   midlvl_label="rollback", repeat_index=0)
        _plot(f"Step 5: Go-back in seg A")

        # 6. Insert on old pass
        ll.pitch_insert(src_time=t_ins, pitch=113, duration=0.2,
                        velocity=70, midlvl_label="mistouch", repeat_index=1)
        _plot(f"Step 6: Insert pitch 113 on OLD pass of seg A")

        # 7. Go back in seg B
        t_back_to_b = float(ll.src_na["onset_sec"][SEG_B[0]])
        t_back_from_b = float(ll.src_na["onset_sec"][SEG_B[1]])
        ll.go_back(src_time_to=t_back_to_b, src_time_from=t_back_from_b,
                   midlvl_label="rollback", repeat_index=0)
        _plot(f"Step 7: Go-back in seg B")

        # Assertions
        assert len(ll.repeat_tracker) == 2
        assert 100 in ll.tgt_na["pitch"]
        assert 113 in ll.tgt_na["pitch"]
        assert any(ll.label_na["lowlvl_label"] == "pitch_insert")
        assert any(ll.label_na["lowlvl_label"] == "time_shift")
        assert any(ll.label_na["lowlvl_label"] == "pitch_delete")
        assert any(ll.label_na["lowlvl_label"] == "change_offset")

        onsets = ll.tgt_na["onset_sec"]
        assert np.all(onsets[:-1] <= onsets[1:])

        plt.close('all')


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
