"""
Tests for lowlvl.py using kv279_1.mid (Mozart K.279 mvt 1) as source data.
Loads via partitura with pedal_threshold=127 to match the caller pipeline.

Covers: pitch_insert, pitch_delete, time_offset, go_back, change_note_offset,
        repeat_tracker logic, _apply_warping_path_offsets, _repeat_tracker_order,
        get_notes, get_notes_between, _label_note, _shift_labels, get_adjusted_gt.
"""

import numpy as np
import copy
import pytest
import partitura as pt
import os

from piano_synmist.lowlvl import lowlvl, regular_na_fields, label_na_fields, DEFAULT_MID

# ---------------------------------------------------------------------------
# Fixture: load MIDI once via partitura, provide fresh lowlvl per test
# ---------------------------------------------------------------------------

MIDI_PATH = os.path.join(os.path.dirname(__file__), "kv279_1.mid")


@pytest.fixture(scope="session")
def src_na():
    """Session-scoped source note array from the MIDI file."""
    perf = pt.load_performance(MIDI_PATH, pedal_threshold=127)
    na = perf.note_array()
    na.sort(order="onset_sec")
    return na


@pytest.fixture
def ll(src_na):
    """Fresh lowlvl instance for each test."""
    return lowlvl(copy.deepcopy(src_na), mode="runthrough")


@pytest.fixture
def ll_seg(src_na):
    """Fresh segmented-mode lowlvl instance."""
    return lowlvl(copy.deepcopy(src_na), mode="segmented")


# Helpers to pick real pitches/times from the piece
def _pick_note(ll_inst, index=50):
    """Return (onset, pitch, duration) of a note deep enough into the piece."""
    n = ll_inst.src_na[index]
    return float(n["onset_sec"]), int(n["pitch"]), float(n["duration_sec"])


def _pick_two_times(ll_inst, idx_a=100, idx_b=200):
    """Return two onset times where idx_a < idx_b (earlier, later)."""
    return float(ll_inst.src_na[idx_a]["onset_sec"]), float(ll_inst.src_na[idx_b]["onset_sec"])


# ===================================================================
# 1. Initialisation
# ===================================================================

class TestInit:
    def test_tgt_is_deep_copy_of_src(self, ll):
        assert len(ll.tgt_na) == len(ll.src_na)
        ll.tgt_na["pitch"][0] = 999
        assert ll.src_na["pitch"][0] != 999

    def test_time_from_equals_time_to_in_runthrough(self, ll):
        np.testing.assert_array_equal(ll.time_from, ll.time_to)

    def test_segmented_mode_empty_tgt(self, ll_seg):
        assert len(ll_seg.tgt_na) == 0
        assert np.all(ll_seg.time_to == -1)

    def test_repeat_tracker_starts_empty(self, ll):
        assert ll.repeat_tracker == {}

    def test_src_na_is_sorted(self, ll):
        onsets = ll.src_na["onset_sec"]
        assert np.all(onsets[:-1] <= onsets[1:])


# ===================================================================
# 2. pitch_insert
# ===================================================================

class TestPitchInsert:
    def test_basic_insert_adds_note(self, ll):
        onset, _, _ = _pick_note(ll)
        original_len = len(ll.tgt_na)
        ll.pitch_insert(src_time=onset, pitch=100, duration=0.3,
                        velocity=80, midlvl_label="mistouch")
        assert len(ll.tgt_na) == original_len + 1

    def test_inserted_note_has_correct_pitch(self, ll):
        onset, _, _ = _pick_note(ll)
        ll.pitch_insert(src_time=onset, pitch=100, duration=0.3,
                        velocity=80, midlvl_label="mistouch")
        assert 100 in ll.tgt_na["pitch"]

    def test_insert_creates_two_labels(self, ll):
        onset, _, _ = _pick_note(ll)
        ll.pitch_insert(src_time=onset, pitch=100, duration=0.3,
                        velocity=80, midlvl_label="mistouch")
        assert len(ll.label_na) == 2  # mid + low

    def test_label_has_correct_lowlvl(self, ll):
        onset, _, _ = _pick_note(ll)
        ll.pitch_insert(src_time=onset, pitch=100, duration=0.3,
                        velocity=80, midlvl_label="mistouch")
        assert any(ll.label_na["lowlvl_label"] == "pitch_insert")

    def test_tgt_remains_sorted_after_insert(self, ll):
        onset, _, _ = _pick_note(ll)
        ll.pitch_insert(src_time=onset, pitch=100, duration=0.3,
                        velocity=80, midlvl_label="mistouch")
        onsets = ll.tgt_na["onset_sec"]
        assert np.all(onsets[:-1] <= onsets[1:])

    def test_insert_at_beginning(self, ll):
        first_onset = float(ll.src_na["onset_sec"][0])
        original_len = len(ll.tgt_na)
        ll.pitch_insert(src_time=first_onset, pitch=100, duration=0.1,
                        velocity=80, midlvl_label="mistouch")
        assert len(ll.tgt_na) == original_len + 1

    def test_multiple_inserts(self, ll):
        t1, _, _ = _pick_note(ll, index=50)
        t2, _, _ = _pick_note(ll, index=150)
        ll.pitch_insert(src_time=t1, pitch=100, duration=0.3,
                        velocity=80, midlvl_label="mistouch")
        ll.pitch_insert(src_time=t2, pitch=101, duration=0.3,
                        velocity=80, midlvl_label="mistouch")
        assert 100 in ll.tgt_na["pitch"]
        assert 101 in ll.tgt_na["pitch"]

    def test_insert_same_pitch_as_existing(self, ll):
        """Insert a duplicate pitch at the same time — both should coexist."""
        onset, pitch, _ = _pick_note(ll)
        count_before = np.sum(ll.tgt_na["pitch"] == pitch)
        ll.pitch_insert(src_time=onset, pitch=pitch, duration=0.2,
                        velocity=80, midlvl_label="mistouch")
        count_after = np.sum(ll.tgt_na["pitch"] == pitch)
        assert count_after == count_before + 1


# ===================================================================
# 3. pitch_delete
# ===================================================================

class TestPitchDelete:
    def test_basic_delete_removes_note(self, ll):
        onset, pitch, _ = _pick_note(ll, index=80)
        original_len = len(ll.tgt_na)
        ll.pitch_delete(src_time=onset, pitch=pitch, midlvl_label="mistouch")
        assert len(ll.tgt_na) == original_len - 1

    def test_deleted_pitch_gone_at_that_time(self, ll):
        onset, pitch, _ = _pick_note(ll, index=80)
        ll.pitch_delete(src_time=onset, pitch=pitch, midlvl_label="mistouch")
        nearby = ll.tgt_na[np.abs(ll.tgt_na["onset_sec"] - onset) < 0.01]
        assert pitch not in nearby["pitch"]

    def test_delete_creates_label(self, ll):
        onset, pitch, _ = _pick_note(ll, index=80)
        ll.pitch_delete(src_time=onset, pitch=pitch, midlvl_label="mistouch")
        assert any(ll.label_na["lowlvl_label"] == "pitch_delete")

    def test_delete_nonexistent_pitch_is_noop(self, ll):
        onset, _, _ = _pick_note(ll)
        original_len = len(ll.tgt_na)
        ll.pitch_delete(src_time=onset, pitch=127, midlvl_label="mistouch")
        assert len(ll.tgt_na) == original_len

    def test_delete_preserves_other_notes_at_same_time(self, ll):
        """Find a chord location and delete one pitch; others survive."""
        onsets = ll.tgt_na["onset_sec"]
        for i in range(len(onsets) - 1):
            if onsets[i] == onsets[i + 1]:
                t = float(onsets[i])
                p1 = int(ll.tgt_na["pitch"][i])
                p2 = int(ll.tgt_na["pitch"][i + 1])
                break
        else:
            pytest.skip("No chord found in piece")

        ll.pitch_delete(src_time=t, pitch=p1, midlvl_label="mistouch")
        nearby = ll.tgt_na[np.abs(ll.tgt_na["onset_sec"] - t) < 0.01]
        assert p2 in nearby["pitch"]


# ===================================================================
# 4. time_offset (adding silence / temporal shift)
# ===================================================================

class TestTimeOffset:
    def test_positive_offset_shifts_later_notes(self, ll):
        t_early, _, _ = _pick_note(ll, index=100)
        old_late_onset = float(ll.tgt_na["onset_sec"][200])
        ll.time_offset(src_time=t_early, offset_time=0.5, midlvl_label="drag")
        new_late_onset = float(ll.tgt_na["onset_sec"][200])
        assert new_late_onset == pytest.approx(old_late_onset + 0.5, abs=0.15)

    def test_offset_shifts_time_to(self, ll):
        t, _, _ = _pick_note(ll, index=100)
        idx_later = np.fabs(ll.time_from - float(ll.src_na["onset_sec"][200])).argmin()
        old_val = float(ll.time_to[idx_later])
        ll.time_offset(src_time=t, offset_time=1.0, midlvl_label="drag")
        assert ll.time_to[idx_later] == pytest.approx(old_val + 1.0, abs=0.15)

    def test_offset_creates_label(self, ll):
        t, _, _ = _pick_note(ll, index=100)
        ll.time_offset(src_time=t, offset_time=0.5, midlvl_label="drag")
        assert any(ll.label_na["lowlvl_label"] == "time_shift")

    def test_negative_offset(self, ll):
        t, _, _ = _pick_note(ll, index=200)
        original_len = len(ll.tgt_na)
        ll.time_offset(src_time=t, offset_time=-0.3, midlvl_label="drag")
        assert len(ll.tgt_na) == original_len

    def test_early_notes_unaffected(self, ll):
        t, _, _ = _pick_note(ll, index=100)
        old_first = float(ll.tgt_na["onset_sec"][0])
        ll.time_offset(src_time=t, offset_time=2.0, midlvl_label="drag")
        assert ll.tgt_na["onset_sec"][0] == pytest.approx(old_first, abs=0.01)


# ===================================================================
# 5. change_note_offset (duration change)
# ===================================================================

class TestChangeNoteOffset:
    def test_extend_note(self, ll):
        onset, pitch, dur = _pick_note(ll, index=80)
        result = ll.change_note_offset(src_time=onset, pitch=pitch,
                                       offset_shift=0.2, midlvl_label="drag")
        assert result is True
        idx = np.where((np.abs(ll.tgt_na["onset_sec"] - onset) < 0.01) &
                       (ll.tgt_na["pitch"] == pitch))[0]
        assert len(idx) > 0
        assert ll.tgt_na["duration_sec"][idx[0]] == pytest.approx(dur + 0.2, abs=0.01)

    def test_shorten_note(self, ll):
        onset, pitch, dur = _pick_note(ll, index=80)
        result = ll.change_note_offset(src_time=onset, pitch=pitch,
                                       offset_shift=-0.02, midlvl_label="drag")
        assert result is True
        idx = np.where((np.abs(ll.tgt_na["onset_sec"] - onset) < 0.01) &
                       (ll.tgt_na["pitch"] == pitch))[0]
        assert ll.tgt_na["duration_sec"][idx[0]] == pytest.approx(dur - 0.02, abs=0.01)

    def test_offset_on_missing_pitch_returns_false(self, ll):
        onset, _, _ = _pick_note(ll)
        result = ll.change_note_offset(src_time=onset, pitch=127,
                                       offset_shift=0.2, midlvl_label="drag")
        assert result is False

    def test_creates_label(self, ll):
        onset, pitch, _ = _pick_note(ll, index=80)
        ll.change_note_offset(src_time=onset, pitch=pitch,
                              offset_shift=0.1, midlvl_label="drag")
        assert any(ll.label_na["lowlvl_label"] == "change_offset")


# ===================================================================
# 6. go_back (rollback)
# ===================================================================

class TestGoBack:
    def _do_goback(self, ll, idx_to=100, idx_from=200):
        t_to = float(ll.src_na["onset_sec"][idx_to])
        t_from = float(ll.src_na["onset_sec"][idx_from])
        notes = ll.get_notes_between(t_to, t_from)
        for n in notes:
            n["onset_sec"] -= notes[0]["onset_sec"]
        ll.go_back(src_time_to=t_to, src_time_from=t_from,
                   notes=notes, midlvl_label="rollback")
        return t_to, t_from

    def test_go_back_adds_notes(self, ll):
        original_len = len(ll.tgt_na)
        self._do_goback(ll)
        assert len(ll.tgt_na) > original_len

    def test_go_back_creates_repeat_tracker_entry(self, ll):
        self._do_goback(ll)
        assert len(ll.repeat_tracker) == 1

    def test_repeat_tracker_key_is_tuple(self, ll):
        self._do_goback(ll)
        key = list(ll.repeat_tracker.keys())[0]
        assert isinstance(key, tuple) and len(key) == 2

    def test_repeat_tracker_value_has_two_arrays(self, ll):
        self._do_goback(ll)
        val = list(ll.repeat_tracker.values())[0]
        assert len(val) == 2
        assert isinstance(val[0], np.ndarray)
        assert isinstance(val[1], np.ndarray)

    def test_go_back_shifts_time_to_forward(self, ll):
        later_idx = np.fabs(ll.time_from - float(ll.src_na["onset_sec"][300])).argmin()
        old_time_to = float(ll.time_to[later_idx])
        self._do_goback(ll, idx_to=100, idx_from=200)
        assert ll.time_to[later_idx] > old_time_to

    def test_go_back_creates_time_shift_label(self, ll):
        self._do_goback(ll)
        assert any(ll.label_na["lowlvl_label"] == "time_shift")

    def test_tgt_sorted_after_goback(self, ll):
        self._do_goback(ll)
        onsets = ll.tgt_na["onset_sec"]
        assert np.all(onsets[:-1] <= onsets[1:])


# ===================================================================
# 7. _apply_warping_path_offsets
# ===================================================================

class TestApplyWarpingPathOffsets:
    def test_offsets_time_to_after_threshold(self, ll):
        mid_time = float(ll.src_na["onset_sec"][len(ll.src_na) // 2])
        idx_later = np.fabs(ll.time_from - mid_time - 5.0).argmin()
        old = float(ll.time_to[idx_later])
        ll._apply_warping_path_offsets(mid_time, 0.5)
        assert ll.time_to[idx_later] == pytest.approx(old + 0.5, abs=0.05)

    def test_does_not_offset_before_threshold(self, ll):
        mid_time = float(ll.src_na["onset_sec"][len(ll.src_na) // 2])
        idx_before = np.fabs(ll.time_from - 2.0).argmin()
        old = float(ll.time_to[idx_before])
        ll._apply_warping_path_offsets(mid_time, 0.5)
        assert ll.time_to[idx_before] == pytest.approx(old, abs=0.01)

    def test_with_repeat_tracker_shifts_values(self, ll):
        """After fixing the typos, repeat_tracker entries should also shift."""
        mid_time = float(ll.src_na["onset_sec"][len(ll.src_na) // 2])
        idx1 = np.fabs(ll.time_from - (mid_time + 1.0)).argmin()
        idx2 = np.fabs(ll.time_from - (mid_time + 3.0)).argmin()
        time_to_sub = ll.time_to[idx1:idx2 + 1].copy()
        time_from_sub = ll.time_from[idx1:idx2 + 1].copy()
        key = (float(time_to_sub[-1]), float(time_to_sub[0]))
        ll.repeat_tracker[key] = [time_to_sub, time_from_sub]

        old_first = float(time_to_sub[0])
        ll._apply_warping_path_offsets(mid_time, 2.0)

        assert key not in ll.repeat_tracker
        new_val = list(ll.repeat_tracker.values())[0]
        assert float(new_val[0][0]) == pytest.approx(old_first + 2.0, abs=0.05)


# ===================================================================
# 8. _repeat_tracker_order
# ===================================================================

class TestRepeatTrackerOrder:
    def test_empty_tracker(self, ll):
        t, _, _ = _pick_note(ll, index=50)
        keys_incl, keys_all = ll._repeat_tracker_order(t)
        assert len(keys_all) == 0
        assert len(keys_incl) == 0

    def test_returns_matching_keys(self, ll):
        """After a go_back, the source time should appear in repeat_tracker_order."""
        t_to = float(ll.src_na["onset_sec"][100])
        t_from = float(ll.src_na["onset_sec"][200])
        notes = ll.get_notes_between(t_to, t_from)
        for n in notes:
            n["onset_sec"] -= notes[0]["onset_sec"]
        ll.go_back(src_time_to=t_to, src_time_from=t_from,
                   notes=notes, midlvl_label="rollback")

        mid_src = float(ll.src_na["onset_sec"][150])
        keys_incl, keys_all = ll._repeat_tracker_order(mid_src)
        assert len(keys_incl) >= 1

    def test_keys_all_is_sorted(self, ll):
        """All keys should be sorted by their tuple values."""
        for start_idx, end_idx in [(100, 150), (300, 350)]:
            idx1 = np.fabs(ll.time_from - float(ll.src_na["onset_sec"][start_idx])).argmin()
            idx2 = np.fabs(ll.time_from - float(ll.src_na["onset_sec"][end_idx])).argmin()
            tt = ll.time_to[idx1:idx2 + 1].copy()
            tf = ll.time_from[idx1:idx2 + 1].copy()
            ll.repeat_tracker[(float(tt[-1]), float(tt[0]))] = (tt, tf)

        _, keys_all = ll._repeat_tracker_order(float(ll.src_na["onset_sec"][120]))
        assert keys_all == sorted(keys_all)


# ===================================================================
# 9. get_notes_between
# ===================================================================

class TestGetNotesBetween:
    def test_returns_notes_in_range(self, ll):
        t1 = float(ll.src_na["onset_sec"][50])
        t2 = float(ll.src_na["onset_sec"][60])
        notes = ll.get_notes_between(t1, t2)
        assert len(notes) >= 10
        assert np.all(notes["onset_sec"] >= t1 - 0.01)
        assert np.all(notes["onset_sec"] <= t2 + 0.01)

    def test_single_time_returns_at_least_one(self, ll):
        t = float(ll.src_na["onset_sec"][50])
        notes = ll.get_notes_between(t, t)
        assert len(notes) >= 1

    def test_full_range(self, ll):
        t1 = float(ll.src_na["onset_sec"][0])
        t2 = float(ll.src_na["onset_sec"][-1])
        notes = ll.get_notes_between(t1, t2)
        assert len(notes) == len(ll.src_na)

    def test_returned_notes_are_copies(self, ll):
        t1 = float(ll.src_na["onset_sec"][50])
        t2 = float(ll.src_na["onset_sec"][55])
        notes = ll.get_notes_between(t1, t2)
        notes["pitch"][0] = 999
        assert ll.src_na["pitch"][50] != 999


# ===================================================================
# 10. get_notes
# ===================================================================

class TestGetNotes:
    def test_zero_events_back(self, ll):
        onset, pitch, _ = _pick_note(ll, index=100)
        idx, notes = ll.get_notes(src_time=onset, num_events_back=0)
        assert pitch in notes["pitch"]

    def test_one_event_back(self, ll):
        onset_curr = float(ll.src_na["onset_sec"][100])
        onset_prev = float(ll.src_na["onset_sec"][99])
        _, notes = ll.get_notes(src_time=onset_curr, num_events_back=1)
        assert any(np.isclose(notes["onset_sec"], onset_curr, atol=0.01))
        assert any(np.isclose(notes["onset_sec"], onset_prev, atol=0.01))

    def test_more_events_than_available(self, ll):
        onset = float(ll.src_na["onset_sec"][5])
        idx, notes = ll.get_notes(src_time=onset, num_events_back=1000)
        assert idx == 0

    def test_chord_counts_as_one_event(self, ll):
        """Find a chord in the real piece and verify it counts as one event."""
        onsets = ll.src_na["onset_sec"]
        chord_idx = None
        for i in range(1, len(onsets)):
            if onsets[i] == onsets[i - 1]:
                chord_idx = i
                break
        if chord_idx is None:
            pytest.skip("No chord found")

        after_chord = chord_idx
        while after_chord < len(onsets) and onsets[after_chord] == onsets[chord_idx]:
            after_chord += 1
        if after_chord >= len(onsets):
            pytest.skip("Chord at end of piece")

        _, notes = ll.get_notes(src_time=float(onsets[after_chord]), num_events_back=1)
        assert len(notes) >= 3


# ===================================================================
# 11. _label_note
# ===================================================================

class TestLabelNote:
    def test_creates_two_label_entries(self, ll):
        ll._label_note(5.0, 5.5, "pitch_insert", "mistouch")
        assert len(ll.label_na) == 2

    def test_label_onset_and_duration(self, ll):
        ll._label_note(5.0, 5.5, "pitch_insert", "mistouch")
        assert all(np.isclose(ll.label_na["onset_sec"], 5.0, atol=0.001))
        assert all(np.isclose(ll.label_na["duration_sec"], 0.5, atol=0.001))

    def test_unknown_midlvl_gets_default_pitch(self, ll):
        ll._label_note(5.0, 5.5, "pitch_insert", "unknown_xyz")
        assert any(ll.label_na["pitch"] == DEFAULT_MID)

    def test_labels_remain_sorted_after_multiple(self, ll):
        ll._label_note(10.0, 10.5, "pitch_insert", "mistouch")
        ll._label_note(5.0, 5.5, "pitch_delete", "mistouch")
        ll._label_note(7.0, 7.3, "change_offset", "drag")
        onsets = ll.label_na["onset_sec"]
        assert np.all(onsets[:-1] <= onsets[1:])


# ===================================================================
# 12. _shift_labels
# ===================================================================

class TestShiftLabels:
    def test_shifts_labels_after_time(self, ll):
        onset1, _, _ = _pick_note(ll, index=100)
        onset2, _, _ = _pick_note(ll, index=200)
        ll._label_note(onset1, onset1 + 0.5, "pitch_insert", "mistouch")
        ll._label_note(onset2, onset2 + 0.5, "pitch_insert", "mistouch")

        old_late = ll.label_na["onset_sec"][ll.label_na["onset_sec"] >= onset2].copy()
        ll._shift_labels(src_time=onset1 + 0.1, offset=1.0)

        new_late = ll.label_na["onset_sec"][ll.label_na["onset_sec"] >= onset2]
        assert np.any(new_late > old_late[0])

    def test_shift_does_not_affect_earlier_labels(self, ll):
        onset1, _, _ = _pick_note(ll, index=50)
        onset2, _, _ = _pick_note(ll, index=200)
        ll._label_note(onset1, onset1 + 0.5, "pitch_insert", "mistouch")
        ll._label_note(onset2, onset2 + 0.5, "pitch_insert", "mistouch")

        earliest_before = float(ll.label_na["onset_sec"][0])
        ll._shift_labels(src_time=onset2 - 0.1, offset=1.0)
        assert ll.label_na["onset_sec"][0] == pytest.approx(earliest_before, abs=0.01)


# ===================================================================
# 13. _find_note_in_tgt
# ===================================================================

class TestFindNoteInTgt:
    def test_finds_existing_note(self, ll):
        onset, pitch, _ = _pick_note(ll, index=80)
        found, idx, start = ll._find_note_in_tgt(onset, pitch)
        assert found is True
        assert ll.tgt_na["pitch"][idx] == pitch

    def test_not_found_returns_false(self, ll):
        onset, _, _ = _pick_note(ll, index=80)
        found, idx, start = ll._find_note_in_tgt(onset, 127)
        assert found is False
        assert idx is None

    def test_finds_correct_note_in_chord(self, ll):
        onsets = ll.tgt_na["onset_sec"]
        for i in range(len(onsets) - 1):
            if onsets[i] == onsets[i + 1]:
                t = float(onsets[i])
                p = int(ll.tgt_na["pitch"][i + 1])
                found, idx, _ = ll._find_note_in_tgt(t, p)
                assert found is True
                assert ll.tgt_na["pitch"][idx] == p
                return
        pytest.skip("No chord found")


# ===================================================================
# 14. Integration: insert then delete
# ===================================================================

class TestInsertThenDelete:
    def test_insert_then_delete_same_pitch(self, ll):
        onset, _, _ = _pick_note(ll, index=100)
        ll.pitch_insert(src_time=onset, pitch=110, duration=0.3,
                        velocity=80, midlvl_label="mistouch")
        assert 110 in ll.tgt_na["pitch"]
        ll.pitch_delete(src_time=onset, pitch=110, midlvl_label="mistouch")
        nearby = ll.tgt_na[np.abs(ll.tgt_na["onset_sec"] - onset) < 0.1]
        assert 110 not in nearby["pitch"]


# ===================================================================
# 15. Integration: offset then insert
# ===================================================================

class TestOffsetThenInsert:
    def test_offset_then_insert_keeps_sorted(self, ll):
        t1, _, _ = _pick_note(ll, index=100)
        t2, _, _ = _pick_note(ll, index=200)
        ll.time_offset(src_time=t1, offset_time=1.0, midlvl_label="drag")
        ll.pitch_insert(src_time=t2, pitch=110, duration=0.3,
                        velocity=80, midlvl_label="mistouch")
        assert 110 in ll.tgt_na["pitch"]
        onsets = ll.tgt_na["onset_sec"]
        assert np.all(onsets[:-1] <= onsets[1:])


# ===================================================================
# 16. get_adjusted_gt
# ===================================================================

class TestGetAdjustedGt:
    def test_identity_mapping(self, src_na):
        """With no modifications, adjusted gt should equal original annotations."""
        ts = np.linspace(float(src_na["onset_sec"][0]),
                         float(src_na["onset_sec"][-1]), 50)
        ll = lowlvl(copy.deepcopy(src_na), mode="runthrough", ts_annot=ts)
        result = ll.get_adjusted_gt()
        np.testing.assert_allclose(result, ts, atol=0.5)


# ===================================================================
# 17. _create_segmented_practice
# ===================================================================

class TestCreateSegmentedPractice:
    def test_noop_in_runthrough_mode(self, ll):
        ll._create_segmented_practice([(5.0, 10.0)])
        assert len(ll.repeat_tracker) == 0


# ===================================================================
# 18. Accessors
# ===================================================================

class TestAccessors:
    def test_get_timemap_length(self, ll):
        pairs = list(ll.get_timemap())
        assert len(pairs) == len(ll.time_from)

    def test_get_repeats_empty_initially(self, ll):
        assert ll.get_repeats() == {}

    def test_get_repeats_after_goback(self, ll):
        t_to = float(ll.src_na["onset_sec"][100])
        t_from = float(ll.src_na["onset_sec"][150])
        notes = ll.get_notes_between(t_to, t_from)
        for n in notes:
            n["onset_sec"] -= notes[0]["onset_sec"]
        ll.go_back(src_time_to=t_to, src_time_from=t_from,
                   midlvl_label="rollback")
        assert len(ll.get_repeats()) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
