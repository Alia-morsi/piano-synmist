"""
Tests for lowlvl.py segmented-practice mode.

Uses kv279_1.mid (Mozart K.279 mvt 1) as source data.
Loads via partitura with pedal_threshold=127 to match the caller pipeline.

Covers: _create_segmented_practice, time_to mapping for segments,
        operations (insert, delete, offset, duration change, go_back) on
        individual segments, repeat_index > 0 targeting earlier passes,
        multiple go-backs across segments, and mixed realistic scenarios.

Usage:
    pytest test_lowlvl_segmented.py -v
"""

import numpy as np
import copy
import pytest
import partitura as pt
import os

from piano_synmist.lowlvl import lowlvl, regular_na_fields, label_na_fields

MIDI_PATH = os.path.join(os.path.dirname(__file__), "kv279_1.mid")

# ── Segment boundaries (note indices into src_na) ────────────────────
SEG_A = (50, 100)    # src ~6.3s – ~11.0s
SEG_B = (150, 200)   # src ~15.2s – ~20.0s
SEG_C = (250, 300)   # src ~25.5s – ~30.2s
TIME_GAP = 3.0


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def src_na():
    perf = pt.load_performance(MIDI_PATH, pedal_threshold=127)
    na = perf.note_array()
    na.sort(order="onset_sec")
    return na


@pytest.fixture(scope="session")
def segments(src_na):
    """Return segment boundaries as (start_sec, end_sec) tuples."""
    return [
        (float(src_na["onset_sec"][a]), float(src_na["onset_sec"][b]))
        for a, b in [SEG_A, SEG_B, SEG_C]
    ]


def _fresh_seg(src_na, segments):
    """Create a segmented lowlvl instance with 3 segments initialized."""
    ll = lowlvl(copy.deepcopy(src_na), mode="segmented")
    ll._create_segmented_practice(segments, time_gap=TIME_GAP)
    return ll


@pytest.fixture
def seg_ll(src_na, segments):
    return _fresh_seg(src_na, segments)


# ── Helpers ──────────────────────────────────────────────────────────

def _pick_note_in_seg(ll, seg_idx_start, offset=5):
    """Pick a note from a specific segment region of src_na."""
    idx = seg_idx_start + offset
    n = ll.src_na[idx]
    return float(n["onset_sec"]), int(n["pitch"]), float(n["duration_sec"])


def _time_to_at_src(ll, src_time):
    """Look up time_to for a given src_time via the main warping path."""
    idx = np.fabs(ll.time_from - src_time).argmin()
    return float(ll.time_to[idx])


def _tgt_notes_near(ll, tgt_time, window=0.1):
    """Return tgt_na notes within a time window around tgt_time."""
    return ll.tgt_na[np.abs(ll.tgt_na["onset_sec"] - tgt_time) < window]


# ===================================================================
# 1. Segmented Initialisation (before _create_segmented_practice)
# ===================================================================

class TestSegmentedRawInit:
    def test_mode_is_segmented(self, src_na):
        ll = lowlvl(copy.deepcopy(src_na), mode="segmented")
        assert ll.mode == "segmented"

    def test_tgt_na_starts_empty(self, src_na):
        ll = lowlvl(copy.deepcopy(src_na), mode="segmented")
        assert len(ll.tgt_na) == 0

    def test_time_to_all_minus_one(self, src_na):
        ll = lowlvl(copy.deepcopy(src_na), mode="segmented")
        assert np.all(ll.time_to == -1)

    def test_repeat_tracker_empty(self, src_na):
        ll = lowlvl(copy.deepcopy(src_na), mode="segmented")
        assert ll.repeat_tracker == {}


# ===================================================================
# 2. _create_segmented_practice
# ===================================================================

class TestCreateSegmentedPractice:

    def test_single_segment_populates_tgt(self, src_na, segments):
        ll = lowlvl(copy.deepcopy(src_na), mode="segmented")
        ll._create_segmented_practice([segments[0]], time_gap=TIME_GAP)
        assert len(ll.tgt_na) > 0

    def test_single_segment_tgt_starts_near_zero(self, src_na, segments):
        ll = lowlvl(copy.deepcopy(src_na), mode="segmented")
        ll._create_segmented_practice([segments[0]], time_gap=TIME_GAP)
        assert ll.tgt_na["onset_sec"][0] == pytest.approx(0.0, abs=0.01)

    def test_time_to_maps_to_tgt_start(self, src_na, segments):
        """time_to at the segment's src start should be close to where notes land."""
        ll = lowlvl(copy.deepcopy(src_na), mode="segmented")
        ll._create_segmented_practice([segments[0]], time_gap=TIME_GAP)
        seg_src_start = segments[0][0]
        tgt_time = _time_to_at_src(ll, seg_src_start)
        assert tgt_time == pytest.approx(ll.tgt_na["onset_sec"][0], abs=0.1)

    def test_time_to_maps_to_tgt_end(self, src_na, segments):
        ll = lowlvl(copy.deepcopy(src_na), mode="segmented")
        ll._create_segmented_practice([segments[0]], time_gap=TIME_GAP)
        seg_src_end = segments[0][1]
        tgt_time = _time_to_at_src(ll, seg_src_end)
        # Should be close to the last tgt note
        assert tgt_time == pytest.approx(ll.tgt_na["onset_sec"][-1], abs=0.5)

    def test_three_segments_all_present(self, seg_ll, segments):
        """All 3 segments' notes should be in tgt_na."""
        for seg_start, seg_end in segments:
            notes = seg_ll.get_notes_between(seg_start, seg_end)
            for pitch in notes["pitch"][:3]:
                assert pitch in seg_ll.tgt_na["pitch"]

    def test_three_segments_have_gaps(self, seg_ll):
        """Between consecutive segments in tgt_na there should be a gap >= time_gap."""
        onsets = seg_ll.tgt_na["onset_sec"]
        gaps = []
        for i in range(len(onsets) - 1):
            diff = onsets[i + 1] - onsets[i]
            if diff > 1.0:
                gaps.append(diff)
        # Should find at least 2 gaps (between seg A-B and B-C)
        assert len(gaps) >= 2
        for g in gaps:
            assert g >= TIME_GAP * 0.9

    def test_repeat_tracker_empty_after_first_create(self, seg_ll):
        """First pass should not create any repeat_tracker entries."""
        assert len(seg_ll.repeat_tracker) == 0

    def test_time_to_minus_one_outside_segments(self, seg_ll, segments):
        """time_to should remain -1 for src regions not covered by any segment."""
        # Pick a src_time between seg A end and seg B start
        between_time = (segments[0][1] + segments[1][0]) / 2
        idx = np.fabs(seg_ll.time_from - between_time).argmin()
        assert seg_ll.time_to[idx] == -1

    def test_tgt_sorted(self, seg_ll):
        onsets = seg_ll.tgt_na["onset_sec"]
        assert np.all(onsets[:-1] <= onsets[1:])

    def test_noop_in_runthrough_mode(self, src_na, segments):
        ll = lowlvl(copy.deepcopy(src_na), mode="runthrough")
        ll._create_segmented_practice(segments)
        assert len(ll.repeat_tracker) == 0

    def test_time_to_increases_within_segment(self, seg_ll, segments):
        """time_to should increase monotonically within each segment."""
        for seg_start, seg_end in segments:
            idx_s = np.fabs(seg_ll.time_from - seg_start).argmin()
            idx_e = np.fabs(seg_ll.time_from - seg_end).argmin()
            tt_slice = seg_ll.time_to[idx_s:idx_e]
            valid = tt_slice[tt_slice != -1]
            if len(valid) > 1:
                assert np.all(valid[:-1] <= valid[1:])

    def test_second_segment_time_to_offset_correct(self, src_na, segments):
        """Segment B's time_to should reflect its insertion_offset."""
        ll = lowlvl(copy.deepcopy(src_na), mode="segmented")
        ll._create_segmented_practice([segments[0], segments[1]], time_gap=TIME_GAP)

        seg_b_src_start = segments[1][0]
        tt_b = _time_to_at_src(ll, seg_b_src_start)

        # Seg B notes in tgt should start after seg A + gap
        seg_a_src_end = segments[0][1]
        tt_a_end = _time_to_at_src(ll, seg_a_src_end)
        # tt_b should be after tt_a_end + gap (approximately)
        assert tt_b > tt_a_end + TIME_GAP * 0.5


# ===================================================================
# 3. pitch_insert on segments (repeat_index=0)
# ===================================================================

class TestSegmentedPitchInsert:
    def test_insert_on_seg_a(self, seg_ll):
        src_time, _, _ = _pick_note_in_seg(seg_ll, SEG_A[0], offset=5)
        original_len = len(seg_ll.tgt_na)
        seg_ll.pitch_insert(src_time=src_time, pitch=100, duration=0.2,
                            velocity=80, midlvl_label="mistouch", repeat_index=0)
        assert len(seg_ll.tgt_na) == original_len + 1
        assert 100 in seg_ll.tgt_na["pitch"]

    def test_insert_lands_at_correct_tgt_time(self, seg_ll):
        """Inserted note should appear at the time_to corresponding to src_time."""
        src_time, _, _ = _pick_note_in_seg(seg_ll, SEG_A[0], offset=10)
        expected_tgt_time = _time_to_at_src(seg_ll, src_time)
        seg_ll.pitch_insert(src_time=src_time, pitch=101, duration=0.2,
                            velocity=80, midlvl_label="mistouch", repeat_index=0)
        inserted = seg_ll.tgt_na[seg_ll.tgt_na["pitch"] == 101]
        assert len(inserted) == 1
        assert inserted["onset_sec"][0] == pytest.approx(expected_tgt_time, abs=0.2)

    def test_insert_on_seg_b(self, seg_ll):
        src_time, _, _ = _pick_note_in_seg(seg_ll, SEG_B[0], offset=5)
        seg_ll.pitch_insert(src_time=src_time, pitch=102, duration=0.2,
                            velocity=80, midlvl_label="mistouch", repeat_index=0)
        assert 102 in seg_ll.tgt_na["pitch"]

    def test_insert_creates_label(self, seg_ll):
        src_time, _, _ = _pick_note_in_seg(seg_ll, SEG_C[0], offset=5)
        seg_ll.pitch_insert(src_time=src_time, pitch=103, duration=0.2,
                            velocity=80, midlvl_label="mistouch", repeat_index=0)
        assert any(seg_ll.label_na["lowlvl_label"] == "pitch_insert")

    def test_tgt_remains_sorted(self, seg_ll):
        src_time, _, _ = _pick_note_in_seg(seg_ll, SEG_B[0], offset=8)
        seg_ll.pitch_insert(src_time=src_time, pitch=104, duration=0.2,
                            velocity=80, midlvl_label="mistouch", repeat_index=0)
        onsets = seg_ll.tgt_na["onset_sec"]
        assert np.all(onsets[:-1] <= onsets[1:])


# ===================================================================
# 4. pitch_delete on segments
# ===================================================================

class TestSegmentedPitchDelete:
    def test_delete_from_seg_a(self, seg_ll):
        src_time, pitch, _ = _pick_note_in_seg(seg_ll, SEG_A[0], offset=5)
        original_len = len(seg_ll.tgt_na)
        seg_ll.pitch_delete(src_time=src_time, pitch=pitch,
                            midlvl_label="mistouch", repeat_index=0)
        assert len(seg_ll.tgt_na) == original_len - 1

    def test_delete_from_seg_b(self, seg_ll):
        src_time, pitch, _ = _pick_note_in_seg(seg_ll, SEG_B[0], offset=5)
        tgt_time = _time_to_at_src(seg_ll, src_time)
        seg_ll.pitch_delete(src_time=src_time, pitch=pitch,
                            midlvl_label="mistouch", repeat_index=0)
        nearby = _tgt_notes_near(seg_ll, tgt_time, window=0.1)
        assert pitch not in nearby["pitch"]

    def test_delete_creates_label(self, seg_ll):
        src_time, pitch, _ = _pick_note_in_seg(seg_ll, SEG_A[0], offset=10)
        seg_ll.pitch_delete(src_time=src_time, pitch=pitch,
                            midlvl_label="mistouch", repeat_index=0)
        assert any(seg_ll.label_na["lowlvl_label"] == "pitch_delete")


# ===================================================================
# 5. time_offset on segments
# ===================================================================

class TestSegmentedTimeOffset:
    def test_offset_shifts_later_notes_in_segment(self, seg_ll):
        """Offset applied in seg A should shift later tgt notes."""
        src_time, _, _ = _pick_note_in_seg(seg_ll, SEG_A[0], offset=3)
        # Pick a later note in the same segment
        src_time_later, _, _ = _pick_note_in_seg(seg_ll, SEG_A[0], offset=40)
        old_tgt_later = _time_to_at_src(seg_ll, src_time_later)

        seg_ll.time_offset(src_time=src_time, offset_time=0.5,
                           midlvl_label="drag", repeat_index=0)

        new_tgt_later = _time_to_at_src(seg_ll, src_time_later)
        assert new_tgt_later == pytest.approx(old_tgt_later + 0.5, abs=0.15)

    def test_offset_in_seg_a_shifts_seg_b(self, seg_ll, segments):
        """Offset in seg A should also shift seg B's time_to (global offset)."""
        src_time_a, _, _ = _pick_note_in_seg(seg_ll, SEG_A[0], offset=5)
        src_time_b, _, _ = _pick_note_in_seg(seg_ll, SEG_B[0], offset=5)
        old_tgt_b = _time_to_at_src(seg_ll, src_time_b)

        seg_ll.time_offset(src_time=src_time_a, offset_time=1.0,
                           midlvl_label="drag", repeat_index=0)

        new_tgt_b = _time_to_at_src(seg_ll, src_time_b)
        assert new_tgt_b == pytest.approx(old_tgt_b + 1.0, abs=0.15)

    def test_offset_creates_label(self, seg_ll):
        src_time, _, _ = _pick_note_in_seg(seg_ll, SEG_B[0], offset=5)
        seg_ll.time_offset(src_time=src_time, offset_time=0.3,
                           midlvl_label="drag", repeat_index=0)
        assert any(seg_ll.label_na["lowlvl_label"] == "time_shift")


# ===================================================================
# 6. change_note_offset on segments
# ===================================================================

class TestSegmentedChangeNoteOffset:
    def test_extend_note_in_seg(self, seg_ll):
        src_time, pitch, dur = _pick_note_in_seg(seg_ll, SEG_A[0], offset=10)
        result = seg_ll.change_note_offset(src_time=src_time, pitch=pitch,
                                           offset_shift=0.2, midlvl_label="drag",
                                           repeat_index=0)
        assert result is True

    def test_extended_duration_correct(self, seg_ll):
        src_time, pitch, dur = _pick_note_in_seg(seg_ll, SEG_B[0], offset=10)
        seg_ll.change_note_offset(src_time=src_time, pitch=pitch,
                                  offset_shift=0.15, midlvl_label="drag",
                                  repeat_index=0)
        tgt_time = _time_to_at_src(seg_ll, src_time)
        found = seg_ll.tgt_na[(np.abs(seg_ll.tgt_na["onset_sec"] - tgt_time) < 0.1) &
                              (seg_ll.tgt_na["pitch"] == pitch)]
        assert len(found) >= 1
        assert found["duration_sec"][0] == pytest.approx(dur + 0.15, abs=0.02)

    def test_creates_label(self, seg_ll):
        src_time, pitch, _ = _pick_note_in_seg(seg_ll, SEG_C[0], offset=10)
        seg_ll.change_note_offset(src_time=src_time, pitch=pitch,
                                  offset_shift=0.1, midlvl_label="drag",
                                  repeat_index=0)
        assert any(seg_ll.label_na["lowlvl_label"] == "change_offset")


# ===================================================================
# 7. go_back within a segment
# ===================================================================

class TestSegmentedGoBack:

    def _do_goback_in_seg(self, ll, seg_start_idx, seg_end_idx):
        """Go back from end to start of a segment."""
        src_time_to = float(ll.src_na["onset_sec"][seg_start_idx])
        src_time_from = float(ll.src_na["onset_sec"][seg_end_idx])
        ll.go_back(src_time_to=src_time_to, src_time_from=src_time_from,
                   midlvl_label="rollback", repeat_index=0)
        return src_time_to, src_time_from

    def test_goback_creates_repeat_tracker(self, seg_ll):
        self._do_goback_in_seg(seg_ll, SEG_A[0], SEG_A[1])
        assert len(seg_ll.repeat_tracker) == 1

    def test_goback_adds_notes(self, seg_ll):
        original_len = len(seg_ll.tgt_na)
        self._do_goback_in_seg(seg_ll, SEG_A[0], SEG_A[1])
        assert len(seg_ll.tgt_na) > original_len

    def test_tgt_sorted_after_goback(self, seg_ll):
        self._do_goback_in_seg(seg_ll, SEG_A[0], SEG_A[1])
        onsets = seg_ll.tgt_na["onset_sec"]
        assert np.all(onsets[:-1] <= onsets[1:])

    def test_repeat_tracker_stores_old_pass(self, seg_ll, segments):
        """The repeat_tracker entry should cover the segment's src time range."""
        self._do_goback_in_seg(seg_ll, SEG_A[0], SEG_A[1])
        val = list(seg_ll.repeat_tracker.values())[0]
        rt_time_to, rt_time_from = val
        # rt_time_from should span the segment's src time range
        assert float(rt_time_from[0]) == pytest.approx(segments[0][0], abs=0.5)
        assert float(rt_time_from[-1]) == pytest.approx(segments[0][1], abs=0.5)

    def test_goback_shifts_time_to_forward(self, seg_ll, segments):
        """After go_back, time_to for later points should increase."""
        src_time_later, _, _ = _pick_note_in_seg(seg_ll, SEG_B[0], offset=5)
        old_tgt = _time_to_at_src(seg_ll, src_time_later)
        self._do_goback_in_seg(seg_ll, SEG_A[0], SEG_A[1])
        new_tgt = _time_to_at_src(seg_ll, src_time_later)
        assert new_tgt > old_tgt

    def test_goback_in_seg_b(self, seg_ll):
        """Go_back in seg B independently."""
        self._do_goback_in_seg(seg_ll, SEG_B[0], SEG_B[1])
        assert len(seg_ll.repeat_tracker) == 1

    def test_goback_creates_time_shift_label(self, seg_ll):
        self._do_goback_in_seg(seg_ll, SEG_A[0], SEG_A[1])
        assert any(seg_ll.label_na["lowlvl_label"] == "time_shift")


# ===================================================================
# 8. Operations on earlier passes via repeat_index
# ===================================================================

class TestOperationsOnRepeat:
    """After a go_back in seg A, test that repeat_index=1 targets
    the first (old) pass stored in repeat_tracker."""

    def _setup_with_goback(self, src_na, segments):
        ll = _fresh_seg(src_na, segments)
        src_time_to = float(ll.src_na["onset_sec"][SEG_A[0]])
        src_time_from = float(ll.src_na["onset_sec"][SEG_A[1]])
        ll.go_back(src_time_to=src_time_to, src_time_from=src_time_from,
                   midlvl_label="rollback", repeat_index=0)
        return ll

    def test_insert_on_old_pass(self, src_na, segments):
        ll = self._setup_with_goback(src_na, segments)
        src_time, _, _ = _pick_note_in_seg(ll, SEG_A[0], offset=10)
        original_len = len(ll.tgt_na)
        ll.pitch_insert(src_time=src_time, pitch=110, duration=0.2,
                        velocity=80, midlvl_label="mistouch", repeat_index=1)
        assert len(ll.tgt_na) == original_len + 1
        assert 110 in ll.tgt_na["pitch"]

    def test_insert_on_old_pass_resolves_through_repeat_tracker(self, src_na, segments):
        """Insert with repeat_index=1 should resolve the tgt time via
        repeat_tracker (not the main time_to). After go_back, both
        mappings point to the same tgt region for the segment, but the
        lookup path through repeat_tracker is confirmed by successfully
        placing the note at the expected tgt time."""
        ll = self._setup_with_goback(src_na, segments)
        src_time, _, _ = _pick_note_in_seg(ll, SEG_A[0], offset=10)

        # Get expected tgt time from repeat_tracker directly
        rt_val = list(ll.repeat_tracker.values())[0]
        rt_time_to, rt_time_from = rt_val
        rt_idx = np.fabs(rt_time_from - src_time).argmin()
        expected_tgt = float(rt_time_to[rt_idx])

        ll.pitch_insert(src_time=src_time, pitch=111, duration=0.2,
                        velocity=80, midlvl_label="mistouch", repeat_index=1)
        inserted = ll.tgt_na[ll.tgt_na["pitch"] == 111]
        assert len(inserted) == 1
        assert inserted["onset_sec"][0] == pytest.approx(expected_tgt, abs=0.3)

    def test_delete_on_old_pass(self, src_na, segments):
        ll = self._setup_with_goback(src_na, segments)
        src_time, pitch, _ = _pick_note_in_seg(ll, SEG_A[0], offset=10)
        original_len = len(ll.tgt_na)
        ll.pitch_delete(src_time=src_time, pitch=pitch,
                        midlvl_label="mistouch", repeat_index=1)
        assert len(ll.tgt_na) == original_len - 1

    def test_time_offset_on_old_pass(self, src_na, segments):
        ll = self._setup_with_goback(src_na, segments)
        src_time, _, _ = _pick_note_in_seg(ll, SEG_A[0], offset=5)

        # Record old pass time_to for a later point
        rt_val = list(ll.repeat_tracker.values())[0]
        rt_time_to, rt_time_from = rt_val
        later_src, _, _ = _pick_note_in_seg(ll, SEG_A[0], offset=30)
        rt_idx_later = np.fabs(rt_time_from - later_src).argmin()
        old_rt_tgt = float(rt_time_to[rt_idx_later])

        ll.time_offset(src_time=src_time, offset_time=0.5,
                       midlvl_label="drag", repeat_index=1)

        # The repeat_tracker values should have shifted
        rt_val_after = list(ll.repeat_tracker.values())[0]
        rt_time_to_after = rt_val_after[0]
        new_rt_tgt = float(rt_time_to_after[rt_idx_later])
        assert new_rt_tgt == pytest.approx(old_rt_tgt + 0.5, abs=0.15)

    def test_change_offset_on_old_pass(self, src_na, segments):
        ll = self._setup_with_goback(src_na, segments)
        src_time, pitch, dur = _pick_note_in_seg(ll, SEG_A[0], offset=10)
        result = ll.change_note_offset(src_time=src_time, pitch=pitch,
                                       offset_shift=0.2, midlvl_label="drag",
                                       repeat_index=1)
        assert result is True

    def test_invalid_repeat_index_raises(self, src_na, segments):
        """repeat_index=2 with only 1 repeat_tracker entry should raise IndexError."""
        ll = self._setup_with_goback(src_na, segments)
        src_time, _, _ = _pick_note_in_seg(ll, SEG_A[0], offset=10)
        with pytest.raises(IndexError):
            ll.pitch_insert(src_time=src_time, pitch=120, duration=0.2,
                            velocity=80, midlvl_label="mistouch", repeat_index=2)

    def test_repeat_index_0_still_targets_current(self, src_na, segments):
        """After go_back, repeat_index=0 should still target the current (new) pass."""
        ll = self._setup_with_goback(src_na, segments)
        src_time, _, _ = _pick_note_in_seg(ll, SEG_A[0], offset=10)
        current_tgt = _time_to_at_src(ll, src_time)

        ll.pitch_insert(src_time=src_time, pitch=112, duration=0.2,
                        velocity=80, midlvl_label="mistouch", repeat_index=0)
        inserted = ll.tgt_na[ll.tgt_na["pitch"] == 112]
        assert len(inserted) == 1
        assert inserted["onset_sec"][0] == pytest.approx(current_tgt, abs=0.2)


# ===================================================================
# 9. Multiple go-backs
# ===================================================================

class TestMultipleGobacks:

    def test_two_gobacks_same_segment(self, src_na, segments):
        """Two go-backs in segment A should create 2 repeat_tracker entries."""
        ll = _fresh_seg(src_na, segments)

        # First go-back (full segment)
        t_to = float(ll.src_na["onset_sec"][SEG_A[0]])
        t_from = float(ll.src_na["onset_sec"][SEG_A[1]])
        ll.go_back(src_time_to=t_to, src_time_from=t_from,
                   midlvl_label="rollback", repeat_index=0)
        assert len(ll.repeat_tracker) == 1

        # Second go-back on the same region
        ll.go_back(src_time_to=t_to, src_time_from=t_from,
                   midlvl_label="rollback", repeat_index=0)
        assert len(ll.repeat_tracker) == 2

    def test_gobacks_in_different_segments(self, src_na, segments):
        """Go-backs in seg A and seg B independently."""
        ll = _fresh_seg(src_na, segments)

        t_to_a = float(ll.src_na["onset_sec"][SEG_A[0]])
        t_from_a = float(ll.src_na["onset_sec"][SEG_A[1]])
        ll.go_back(src_time_to=t_to_a, src_time_from=t_from_a,
                   midlvl_label="rollback", repeat_index=0)

        t_to_b = float(ll.src_na["onset_sec"][SEG_B[0]])
        t_from_b = float(ll.src_na["onset_sec"][SEG_B[1]])
        ll.go_back(src_time_to=t_to_b, src_time_from=t_from_b,
                   midlvl_label="rollback", repeat_index=0)

        assert len(ll.repeat_tracker) == 2

    def test_repeat_tracker_order_after_two_gobacks(self, src_na, segments):
        """After 2 go-backs in seg A, _repeat_tracker_order should find 2 entries
        for a src time within seg A."""
        ll = _fresh_seg(src_na, segments)
        t_to = float(ll.src_na["onset_sec"][SEG_A[0]])
        t_from = float(ll.src_na["onset_sec"][SEG_A[1]])

        ll.go_back(src_time_to=t_to, src_time_from=t_from,
                   midlvl_label="rollback", repeat_index=0)
        ll.go_back(src_time_to=t_to, src_time_from=t_from,
                   midlvl_label="rollback", repeat_index=0)

        mid_src = float(ll.src_na["onset_sec"][(SEG_A[0] + SEG_A[1]) // 2])
        keys_incl, _ = ll._repeat_tracker_order(mid_src)
        assert len(keys_incl) == 2

    def test_operations_on_second_repeat(self, src_na, segments):
        """After 2 go-backs in seg A, repeat_index=2 should target the second
        (more recent stored) pass."""
        ll = _fresh_seg(src_na, segments)
        t_to = float(ll.src_na["onset_sec"][SEG_A[0]])
        t_from = float(ll.src_na["onset_sec"][SEG_A[1]])

        ll.go_back(src_time_to=t_to, src_time_from=t_from,
                   midlvl_label="rollback", repeat_index=0)
        ll.go_back(src_time_to=t_to, src_time_from=t_from,
                   midlvl_label="rollback", repeat_index=0)

        src_time, _, _ = _pick_note_in_seg(ll, SEG_A[0], offset=10)
        original_len = len(ll.tgt_na)
        ll.pitch_insert(src_time=src_time, pitch=115, duration=0.2,
                        velocity=80, midlvl_label="mistouch", repeat_index=2)
        assert len(ll.tgt_na) == original_len + 1
        assert 115 in ll.tgt_na["pitch"]


# ===================================================================
# 10. Mixed realistic scenario
# ===================================================================

class TestSegmentedMixed:
    def test_full_scenario(self, src_na, segments):
        """
        Realistic sequence:
          1. Create segmented practice with 3 segments
          2. Insert mistouch in seg A
          3. Add hesitation (time_offset) in seg A
          4. Delete a note from seg B
          5. Go back in seg A → creates repeat_tracker entry
          6. Insert on old pass (repeat_index=1)
          7. Change duration on seg C
        """
        ll = _fresh_seg(src_na, segments)

        # 1. Insert mistouch in seg A
        t_ins, _, _ = _pick_note_in_seg(ll, SEG_A[0], offset=8)
        ll.pitch_insert(src_time=t_ins, pitch=100, duration=0.15,
                        velocity=80, midlvl_label="mistouch", repeat_index=0)
        assert 100 in ll.tgt_na["pitch"]

        # 2. Hesitation in seg A
        t_off, _, _ = _pick_note_in_seg(ll, SEG_A[0], offset=15)
        ll.time_offset(src_time=t_off, offset_time=0.6,
                       midlvl_label="drag", repeat_index=0)

        # 3. Delete note from seg B
        t_del, p_del, _ = _pick_note_in_seg(ll, SEG_B[0], offset=10)
        original_len = len(ll.tgt_na)
        ll.pitch_delete(src_time=t_del, pitch=p_del,
                        midlvl_label="mistouch", repeat_index=0)
        assert len(ll.tgt_na) == original_len - 1

        # 4. Go back in seg A
        t_back_to = float(ll.src_na["onset_sec"][SEG_A[0]])
        t_back_from = float(ll.src_na["onset_sec"][SEG_A[1]])
        ll.go_back(src_time_to=t_back_to, src_time_from=t_back_from,
                   midlvl_label="rollback", repeat_index=0)
        assert len(ll.repeat_tracker) == 1

        # 5. Insert on old pass
        ll.pitch_insert(src_time=t_ins, pitch=113, duration=0.2,
                        velocity=70, midlvl_label="mistouch", repeat_index=1)
        assert 113 in ll.tgt_na["pitch"]

        # 6. Change duration on seg C
        t_dur, p_dur, _ = _pick_note_in_seg(ll, SEG_C[0], offset=10)
        ll.change_note_offset(src_time=t_dur, pitch=p_dur,
                              offset_shift=0.25, midlvl_label="drag",
                              repeat_index=0)

        # Verify labels accumulated
        assert any(ll.label_na["lowlvl_label"] == "pitch_insert")
        assert any(ll.label_na["lowlvl_label"] == "time_shift")
        assert any(ll.label_na["lowlvl_label"] == "pitch_delete")
        assert any(ll.label_na["lowlvl_label"] == "change_offset")

        # tgt_na still sorted
        onsets = ll.tgt_na["onset_sec"]
        assert np.all(onsets[:-1] <= onsets[1:])

    def test_operations_across_all_segments(self, src_na, segments):
        """Operations on all 3 segments with interleaved go_backs."""
        ll = _fresh_seg(src_na, segments)

        # Insert in each segment
        for seg_start, seg_end in [(SEG_A[0], 100 + 70), (SEG_B[0], 150 + 70), (SEG_C[0], 250 + 70)]:
            src_t, _, _ = _pick_note_in_seg(ll, seg_start, offset=5)
            pitch = 100 + seg_start
            ll.pitch_insert(src_time=src_t, pitch=pitch, duration=0.1,
                            velocity=80, midlvl_label="mistouch", repeat_index=0)

        # Go back in seg A
        t_to_a = float(ll.src_na["onset_sec"][SEG_A[0]])
        t_from_a = float(ll.src_na["onset_sec"][SEG_A[1]])
        ll.go_back(src_time_to=t_to_a, src_time_from=t_from_a,
                   midlvl_label="rollback", repeat_index=0)

        # Go back in seg B
        t_to_b = float(ll.src_na["onset_sec"][SEG_B[0]])
        t_from_b = float(ll.src_na["onset_sec"][SEG_B[1]])
        ll.go_back(src_time_to=t_to_b, src_time_from=t_from_b,
                   midlvl_label="rollback", repeat_index=0)

        assert len(ll.repeat_tracker) == 2

        # Insert on old pass of seg A (repeat_index=1)
        src_t_a, _, _ = _pick_note_in_seg(ll, SEG_A[0], offset=10)
        ll.pitch_insert(src_time=src_t_a, pitch=200, duration=0.1,
                        velocity=80, midlvl_label="mistouch", repeat_index=1)
        assert 200 in ll.tgt_na["pitch"]

        # tgt_na sorted
        onsets = ll.tgt_na["onset_sec"]
        assert np.all(onsets[:-1] <= onsets[1:])


# ===================================================================
# 11. Edge cases
# ===================================================================

class TestSegmentedEdgeCases:

    def test_insert_between_segments_is_noop(self, seg_ll, segments):
        """Inserting at a src_time in the gap between segments should be
        silently skipped (time_to=-1 guard)."""
        between_time = (segments[0][1] + segments[1][0]) / 2
        original_len = len(seg_ll.tgt_na)
        seg_ll.pitch_insert(src_time=between_time, pitch=99, duration=0.1,
                            velocity=80, midlvl_label="mistouch")
        assert len(seg_ll.tgt_na) == original_len
        assert 99 not in seg_ll.tgt_na["pitch"]

    def test_delete_between_segments_is_noop(self, seg_ll, segments):
        between_time = (segments[0][1] + segments[1][0]) / 2
        original_len = len(seg_ll.tgt_na)
        seg_ll.pitch_delete(src_time=between_time, pitch=60,
                            midlvl_label="mistouch")
        assert len(seg_ll.tgt_na) == original_len

    def test_time_offset_between_segments_is_noop(self, seg_ll, segments):
        between_time = (segments[0][1] + segments[1][0]) / 2
        src_in_seg_b, _, _ = _pick_note_in_seg(seg_ll, SEG_B[0], offset=5)
        old_tgt_b = _time_to_at_src(seg_ll, src_in_seg_b)
        seg_ll.time_offset(src_time=between_time, offset_time=5.0,
                           midlvl_label="drag")
        # Nothing should have shifted
        assert _time_to_at_src(seg_ll, src_in_seg_b) == pytest.approx(old_tgt_b, abs=0.01)

    def test_goback_between_segments_is_noop(self, seg_ll, segments):
        between_start = (segments[0][1] + segments[1][0]) / 2
        between_end = between_start + 1.0
        original_rt_len = len(seg_ll.repeat_tracker)
        original_tgt_len = len(seg_ll.tgt_na)
        seg_ll.go_back(src_time_to=between_start, src_time_from=between_end,
                       midlvl_label="rollback")
        assert len(seg_ll.repeat_tracker) == original_rt_len
        assert len(seg_ll.tgt_na) == original_tgt_len

    def test_change_offset_between_segments_returns_false(self, seg_ll, segments):
        between_time = (segments[0][1] + segments[1][0]) / 2
        result = seg_ll.change_note_offset(src_time=between_time, pitch=60,
                                           offset_shift=0.2, midlvl_label="drag")
        assert result is False

    def test_time_to_is_negative_one_between_segments(self, seg_ll, segments):
        between = (segments[0][1] + segments[1][0]) / 2
        idx = np.fabs(seg_ll.time_from - between).argmin()
        assert seg_ll.time_to[idx] == -1

    def test_goback_notes_param_is_overwritten(self, src_na, segments):
        """go_back always re-fetches notes from src_na regardless of the
        notes parameter. This documents the current behavior (line 443)."""
        ll = _fresh_seg(src_na, segments)
        t_to = float(ll.src_na["onset_sec"][SEG_A[0]])
        t_from = float(ll.src_na["onset_sec"][SEG_A[1]])

        # Pass dummy notes — they should be ignored
        dummy_notes = ll.get_notes_between(t_to, t_to)  # just 1 note

        ll.go_back(src_time_to=t_to, src_time_from=t_from,
                   notes=dummy_notes, midlvl_label="rollback")

        assert len(ll.repeat_tracker) == 1


# ===================================================================
# 12. Global offset behavior (known limitation)
# ===================================================================

class TestGlobalOffsetBehavior:
    """_apply_warping_path_offsets shifts ALL time_to values globally
    (both main array and repeat_tracker). This means time_offset on
    one pass also affects the other. These tests document this."""

    def test_offset_on_current_also_shifts_old_pass(self, src_na, segments):
        ll = _fresh_seg(src_na, segments)
        t_to = float(ll.src_na["onset_sec"][SEG_A[0]])
        t_from = float(ll.src_na["onset_sec"][SEG_A[1]])
        ll.go_back(src_time_to=t_to, src_time_from=t_from,
                   midlvl_label="rollback")

        src_time = float(ll.src_na["onset_sec"][SEG_A[0] + 20])
        rt_val = list(ll.repeat_tracker.values())[0]
        rt_tt, rt_tf = rt_val
        rt_idx = np.fabs(rt_tf - src_time).argmin()
        old_rt_tgt = float(rt_tt[rt_idx])

        # Offset on current pass (repeat_index=0)
        ll.time_offset(src_time=src_time, offset_time=1.0,
                       midlvl_label="drag", repeat_index=0)

        rt_val_after = list(ll.repeat_tracker.values())[0]
        new_rt_tgt = float(rt_val_after[0][rt_idx])
        # Old pass also shifted (global offset)
        assert new_rt_tgt == pytest.approx(old_rt_tgt + 1.0, abs=0.15)

    def test_offset_on_old_pass_also_shifts_current(self, src_na, segments):
        ll = _fresh_seg(src_na, segments)
        t_to = float(ll.src_na["onset_sec"][SEG_A[0]])
        t_from = float(ll.src_na["onset_sec"][SEG_A[1]])
        ll.go_back(src_time_to=t_to, src_time_from=t_from,
                   midlvl_label="rollback")

        src_time = float(ll.src_na["onset_sec"][SEG_A[0] + 20])
        idx_main = np.fabs(ll.time_from - src_time).argmin()
        old_main_tgt = float(ll.time_to[idx_main])

        # Offset on old pass (repeat_index=1)
        ll.time_offset(src_time=src_time, offset_time=0.8,
                       midlvl_label="drag", repeat_index=1)

        # Current pass also shifted
        assert ll.time_to[idx_main] == pytest.approx(old_main_tgt + 0.8, abs=0.15)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
