import partitura as pt
import mido
import math
import numpy as np
import pandas as pd
import numpy.lib.recfunctions as rfn
import copy
import lowlvl 
import csv

#Parametrizations:
MAX_DUR4DRAG = 0.5

rollback_association_prob = {
    'mistouch': 0, 
    'pitch_change': 0.5,
    'drag': 0.6
}

def sort_payload(payload):
    def precedence(x):
        ranking = {'forward_backward_insertion': 0, 
                   'mistouch': 1, 
                   'pitch_change': 2,
                   'drag': 3, 
                   'rollback': 4}
        return ranking[x]
    
    payload = sorted(payload, key=lambda x: (x[0], precedence(x[1])))
    return payload

def print_payload_item(p):
    print('{} - {}'.format(p[0], p[1]))
def print_payload_list(payload):
    for p in payload:
        print_payload_item(p)


class Mistaker():
    def __init__(self, performance_path, time_series_annotation=None, mode='runthrough',
                 segments=None, time_gap=3.0, use_region_classifier=True,
                 sampling_prob_path="sampling_prob.csv"):
        """Overlay synthetic mistakes onto a MIDI piano performance.

        Args:
            performance_path (str): path to MIDI performance file
            time_series_annotation: optional time series annotation array
            mode (str): 'runthrough' or 'segmented'
            segments (list): list of (start_sec, end_sec) tuples for segmented mode
            time_gap (float): gap between segments in segmented mode (default 3.0)
            use_region_classifier (bool): whether to use RegionClassifier for texture
                detection.  Set to False for config-based mistake specification
                without texture analysis.
            sampling_prob_path (str): path to sampling probability CSV
        """
        self.performance_path = performance_path
        self.performance = pt.load_performance(performance_path, pedal_threshold=127)
        self.time_series_annotation = time_series_annotation
        self.mode = mode

        na = self.performance.note_array()
        na.sort(order='onset_sec')

        self.change_tracker = lowlvl.lowlvl(
            na, mode=mode,
            ts_annot=time_series_annotation if time_series_annotation is not None else [])
        
        if mode == 'segmented' and segments is not None:
            self.change_tracker._create_segmented_practice(segments, time_gap=time_gap)
            self.segments = segments
        else:
            self.segments = None

        if use_region_classifier:
            from region_classifier import RegionClassifier
            rc = RegionClassifier(performance_path, save=False)
            self.na = rc.na
        else:
            # Add offset_sec if missing (RegionClassifier normally provides it).
            if 'offset_sec' not in na.dtype.names:
                import numpy.lib.recfunctions as rfn
                offset_sec = na['onset_sec'] + na['duration_sec']
                na = rfn.append_fields(na, 'offset_sec', offset_sec, dtypes='<f4')
            self.na = na

        self.white_keys, self.black_keys = self.black_white_keys()

        self.sampling_prob = None
        if use_region_classifier:
            try:
                self.sampling_prob = pd.read_csv(sampling_prob_path, index_col='index')
            except FileNotFoundError:
                pass

    def schedule_mistakes(self):
        self.mistake_scheduler()

    def get_texture_group(self, note):
        texture_group = None
        if note['is_block_chords_note'] == 1:
            texture_group = 'is_block_chords_note'
        elif note['is_scale_note'] == 1:
            texture_group = 'is_scale_note'
        elif note['is_double_note'] == 1:
            texture_group = 'is_double_note'
        elif note['others'] == 1:
            texture_group = 'others'
        return texture_group
    
    def get_mistake_probability(self, texture):
        mask = self.sampling_prob.index == texture
        probability_index = np.where(self.sampling_prob.index == texture)[0][0]
        return self.sampling_prob.iloc[probability_index]

    def black_white_keys(self):
        white_keys_all_octaves = []
        black_keys_all_octaves = []
        for octave in range(0, 11):
            base_midi_number = 12 + octave * 12
            white_keys = [base_midi_number, base_midi_number + 2, base_midi_number + 4,
                        base_midi_number + 5, base_midi_number + 7, base_midi_number + 9,
                        base_midi_number + 11]
            black_keys = [base_midi_number + 1, base_midi_number + 3, base_midi_number + 6,
                        base_midi_number + 8, base_midi_number + 10]
            white_keys_all_octaves.extend(white_keys)
            black_keys_all_octaves.extend(black_keys)
        return white_keys_all_octaves, black_keys_all_octaves
    
    ########## Config-based mistake specification (no sampling / no texture) #######

    def create_payload_from_config(self, config_list):
        """Create a payload from a list of mistake configuration dicts.
        
        This is the non-interactive, programmatic way to specify mistakes.
        Each dict in config_list should have:
            - 'src_time': float, the source time for the mistake
            - 'mistake_type': str, one of 'mistouch', 'pitch_change', 'drag',
                              'rollback', 'forward_backward_insertion'
        
        Optional fields:
            - 'pitch': int, specific pitch (if not given, nearest note is used)
            - 'forward': bool, for forward_backward_insertion (default: random)
            - 'events_back_range': tuple, for rollback (default: (0, 5))
            - 'with_rollback': bool, for pitch_change/drag (default: False)
        
        Returns:
            list: sorted payload ready for apply_payload
        """
        payload = []
        na = self.na

        for item in config_list:
            src_time = item['src_time']
            mistake_type = item['mistake_type']

            # Find the nearest note to the given src_time
            onset_dists = np.abs(na['onset_sec'] - src_time)
            nearest_idx = onset_dists.argmin()
            
            # If a pitch is specified, refine the search
            if 'pitch' in item:
                pitch = item['pitch']
                pitch_mask = na['pitch'] == pitch
                if np.any(pitch_mask):
                    onset_dists_pitched = onset_dists.copy()
                    onset_dists_pitched[~pitch_mask] = np.inf
                    nearest_idx = onset_dists_pitched.argmin()

            note = na[nearest_idx]

            if mistake_type == 'forward_backward_insertion':
                forward = item.get('forward', np.random.random() > 0.5)
                payload.append((note['onset_sec'], 'forward_backward_insertion', 
                               {'note': note, 'forward': forward}))

            elif mistake_type == 'mistouch':
                payload.append((note['onset_sec'], 'mistouch', {'note': note}))

            elif mistake_type == 'pitch_change':
                payload.append((note['onset_sec'], 'pitch_change', {'note': note}))
                if item.get('with_rollback', False):
                    events_back = item.get('events_back_range', (0, 5))
                    payload.append((note['onset_sec'], 'rollback', 
                                   {'note': note, 'events_back_range': events_back}))

            elif mistake_type == 'drag':
                payload.append((note['onset_sec'], 'drag', {'note': note}))
                if item.get('with_rollback', False):
                    events_back = item.get('events_back_range', (0, 5))
                    payload.append((note['onset_sec'], 'rollback', 
                                   {'note': note, 'events_back_range': events_back}))

            elif mistake_type == 'rollback':
                events_back = item.get('events_back_range', (0, 5))
                payload.append((note['onset_sec'], 'rollback', 
                               {'note': note, 'events_back_range': events_back}))
            else:
                print(f"Warning: unknown mistake_type '{mistake_type}', skipping")

        payload = sort_payload(payload)
        return payload

    def create_payload(self, parsed_list):
        """Creates payload from the interactive parsed list of note locations."""
        payload = []
        rollback_dice = np.random.random()

        for mistake_item in parsed_list:
            start_time = mistake_item['start_time']
            end_time = mistake_item['end_time']
            pitch = mistake_item['pitch']
            mistake_type = mistake_item['mistake_type']

            notes_in_range = [note for note in self.na if note['onset_sec'] >= start_time and note['onset_sec'] < end_time]
            note_match = None
            for note in notes_in_range:
                if note['pitch'] == pitch:
                    note_match = note
            if note_match is None: 
                note_match = notes_in_range[np.random.choice(len(notes_in_range), 1, replace=True)]

            note = note_match
            if mistake_type == 'forward_backward_insertion':
                payload.append((note['onset_sec'], 'forward_backward_insertion', {'note': note, 'forward': np.random.random() > 0.5}))
            if mistake_type == 'mistouch':
                payload.append((note['onset_sec'], 'mistouch', {'note': note,}))
            if mistake_type == 'pitch_change':
                payload.append((note['onset_sec'], 'pitch_change', {'note': note,}))
                if rollback_dice < rollback_association_prob['pitch_change']:
                    payload.append((note['onset_sec'], 'rollback', {'note': note, 'events_back_range': (0,5)}))
            if mistake_type == 'drag':
                payload.append((note['onset_sec'], 'drag', {'note': note,}))
                if rollback_dice < rollback_association_prob['drag']:
                   payload.append((note['onset_sec'], 'rollback', {'note': note, 'events_back_range': (0,5)}))
            if mistake_type == 'rollback':
                payload.append((note['onset_sec'], 'rollback', {'note': note, 'events_back_range': (0,5)}))

        payload = sort_payload(payload)
        return payload
    
    def interactive_mistake_locations(self):
        """Interactive function: query notes, input mistakes, quit."""
        def parse_mistake_string(input_string):
            import re
            pattern = re.compile(r"\((\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\)-(\d+)-([\w\-]+)")
            parsed_list = []
            syntax_errors = []
            units = input_string.split(';')
            for i, unit in enumerate(units):
                unit = unit.strip()
                match = pattern.match(unit)
                if match:
                    s, e, midi_pitch, mistake_type = match.groups()
                    parsed_list.append({
                        'start_time': float(s),
                        'end_time': float(e),
                        'pitch': int(midi_pitch),
                        'mistake_type': mistake_type
                    })
                else:
                    syntax_errors.append((i, unit))
            return parsed_list, syntax_errors

        parsed_list = []
        syntax_errors = []
        print("Welcome to the Interactive Mistake Editor!")
        while True:
            print("Options:")
            print("1. Query notes within a time range.")
            print("2. Input mistake data.")
            print("3. Quit.")
            choice = input("\nEnter your choice (1/2/3): ").strip()
            if choice == "1":
                try:
                    start_time = float(input("Enter start time (seconds): ").strip())
                    end_time = float(input("Enter end time (seconds): ").strip())
                    if start_time > end_time:
                        print("Start time must be less than or equal to end time.")
                        continue
                    notes_in_range = [
                        note for note in self.na 
                        if note['onset_sec'] >= start_time and note['onset_sec'] <= end_time
                    ]
                    if notes_in_range:
                        print(f"Notes between {start_time}s and {end_time}s:")
                        for note in notes_in_range:
                            print(note)
                    else:
                        print(f"No notes found between {start_time}s and {end_time}s.")
                except ValueError:
                    print("Invalid input.")
            elif choice == "2":
                print("Enter mistake data in the format: (s, e)-<midi_pitch>-<mistake-type>;")
                print("Type 'done' when finished.")
                while True:
                    user_input = input("Enter mistake data: ").strip()
                    if user_input.lower() == "done":
                        break
                    parsed_data, errors = parse_mistake_string(user_input)
                    parsed_list.extend(parsed_data)
                    syntax_errors.extend(errors)
                    if parsed_data:
                        print(f"Successfully added {len(parsed_data)} mistakes.")
                    if errors:
                        print(f"Encountered {len(errors)} syntax errors.")
                        for idx, error in errors:
                            print(f"  Unit {idx + 1}: {error}")
            elif choice == "3":
                print("\nExiting the Mistake Editor.")
                print("Final Mistake Data:")
                for mistake in parsed_list:
                    print(mistake)
                if syntax_errors:
                    print("\nEncountered syntax errors:")
                    for idx, error in syntax_errors:
                        print(f"  Unit {idx + 1}: {error}")
                print("\nInput complete.")
                break
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")
        return parsed_list, syntax_errors
   
    def save_mistake_locations(self, types_and_locs, path):
        return

    def load_mistake_locations(self, path):
        types_and_locs = []
        return types_and_locs
    
    ########### Function for scheduling mistakes #################
    def mistake_scheduler(self, n_mistakes=80):
        """Generate mistakes by sampling notes based on texture probabilities."""
        if self.sampling_prob is None:
            raise RuntimeError("mistake_scheduler requires sampling probabilities "
                               "(use_region_classifier=True and valid sampling_prob.csv)")
        payload = []
        sampled_notes = [self.sample_note(self.na) for _ in range(n_mistakes)]

        for note in sampled_notes:
            texture_group = self.get_texture_group(note)
            mistake_probabilities = self.get_mistake_probability(texture_group)
            mistake_types, probabilities = zip(*mistake_probabilities.items())
            probabilities = np.array(probabilities) / np.sum(probabilities)
            mistake_type = np.random.choice(mistake_types, p=probabilities)

            rollback_dice = np.random.random()   

            if mistake_type == 'forward_backward_insertion':
                payload.append((note['onset_sec'], 'forward_backward_insertion', {'note': note, 'forward': np.random.random() > 0.5}))
            if mistake_type == 'mistouch':
                payload.append((note['onset_sec'], 'mistouch', {'note': note,}))
            if mistake_type == 'pitch_change':
                payload.append((note['onset_sec'], 'pitch_change', {'note': note,}))
                if rollback_dice < rollback_association_prob['pitch_change']:
                    payload.append((note['onset_sec'], 'rollback', {'note': note, 'events_back_range': (0,5)}))
            if mistake_type == 'drag': 
                payload.append((note['onset_sec'], 'drag', {'note': note,}))
                if rollback_dice < rollback_association_prob['drag']:
                    payload.append((note['onset_sec'], 'rollback', {'note': note, 'events_back_range': (0,10)}))
            # FIX: handle standalone rollback from sampling (was silently dropped before)
            if mistake_type == 'rollback':
                payload.append((note['onset_sec'], 'rollback', {'note': note, 'events_back_range': (0, 5)}))

        payload = sort_payload(payload)
        return payload

    def apply_payload(self, payload):
        for i, p in enumerate(payload):
            try:
                self.__getattribute__(p[1])(**p[2])
            except Exception as e:
                    print(e)
        return payload

    def sample_note(self, data):
        return data[np.random.choice(len(data))]

    def sample_group(self, data, group):
        mask = data[group] == 1
        group_data = data[mask]
        if len(group_data) == 0:
            print(f"no data in group {group}.")
            return group_data
        return group_data[np.random.choice(len(group_data))]

    ########### Mid Level Mistake Functions ############
    def rollback(self, note, events_back_range):
        num_events_back = np.random.randint(events_back_range[0], events_back_range[1])
        idx, notes_to_repeat = self.change_tracker.get_notes(note['onset_sec'], num_events_back)

        onset_shift = notes_to_repeat['onset_sec'][0]
        # NOTE: go_back always re-fetches notes from src_na internally (lowlvl.go_back line 452),
        # so we no longer zero out notes_to_repeat here — that was dead code.

        hesitation = np.random.uniform(0.2, 0.8)
        self.change_tracker.time_offset(note['onset_sec'] + note['duration_sec'], hesitation, 'rollback')
        self.change_tracker.go_back(onset_shift, note['onset_sec'] + note['duration_sec'])
        return

    def forward_backward_insertion(self, note, forward=True, ascending=True):
        """Insert notes that belong to the previous / later onset."""
        if forward:
            insert_pitches = self.na[self.na['onset_sec'] > note['offset_sec']]
            min_onset = insert_pitches['onset_sec'].min()
            insert_pitches = insert_pitches[np.isclose(insert_pitches['onset_sec'], min_onset, atol=0.05)]
        else:
            insert_pitches = self.na[self.na['offset_sec'] < note['onset_sec']]
            max_onset = insert_pitches['onset_sec'].max()
            insert_pitches = insert_pitches[np.isclose(insert_pitches['onset_sec'], max_onset, atol=0.05)]

        if (ascending and forward) or ((not ascending) and (not forward)):
            insert_pitches_ = insert_pitches[insert_pitches['pitch'] > note['pitch']]
        else:
            insert_pitches_ = insert_pitches[insert_pitches['pitch'] < note['pitch']]

        if not len(insert_pitches_):
            insert_pitches_ = insert_pitches
        insert_pitch = np.random.choice(insert_pitches_)
        
        onset = note['onset_sec'][0] + np.random.uniform(low=0.0, high=0.5) * 0.05
        duration = note['duration_sec'][0] + np.random.uniform(low=0.0, high=0.5) * 0.05 
        velocity = int(((np.random.random() * 0.5) + 0.5) * note['velocity'])

        self.change_tracker.pitch_insert(onset, insert_pitch['pitch'], duration, velocity, "fwdbackwd") 
        print(f"added forward={forward} insertion at note {note['id']} with pitch {insert_pitch['pitch']}.")

    def mistouch(self, note):
        """Add mistouched inserted note for the given note."""
        if note['pitch'] in self.white_keys: 
            insert_pitch = self.white_keys[self.white_keys.index(note['pitch']) + (np.random.choice([1, -1]))]
            assert(insert_pitch != note['pitch'])
        else:
            insert_pitch = note['pitch'] + (np.random.choice([1, -1]))

        duration = 0.2
        velocity = np.random.randint(30, 70)

        self.change_tracker.pitch_insert(note['onset_sec'], insert_pitch, duration, velocity, "mistouch")
        print(f"added mistouch insertion at note {note['id']} with pitch {insert_pitch}.")

    def pitch_change(self, note, rollback=False, change_chordblock=False):
        """Change the pitch of the given note."""
        changed_pitch = note['pitch']
        if np.random.random() > 0.5:
            neighbor_pitches = self.na[self.na['offset_sec'] < note['onset_sec']]
            if len(neighbor_pitches):
                max_onset = neighbor_pitches['onset_sec'].max()
                neighbor_pitches = neighbor_pitches[np.isclose(neighbor_pitches['onset_sec'], max_onset, atol=0.05)]
                near_neighbor = neighbor_pitches[np.abs(neighbor_pitches['pitch'] - note['pitch']).argmin()]
                changed_pitch = near_neighbor['pitch']
            else:
                changed_pitch = note['pitch'] + np.random.choice([-2, -1, 1, 2])
        
        self.change_tracker.pitch_insert(note['onset_sec'], changed_pitch, note['duration_sec'], note['velocity'], "wrong_pred")
        self.change_tracker.pitch_delete(note['onset_sec'], note['pitch'], "wrong_pred")
        print(f"added pitch change at note {note['id']} with pitch {changed_pitch}.")

    def drag(self, note, drag_window=5):
        """A drag / hesitation on the note position."""
        notes_shortly_after = [n for n in self.performance.performedparts[0].notes 
                               if 0 < n['note_on'] - note['onset_sec'] <= min(note['duration_sec'], MAX_DUR4DRAG) * drag_window]

        notes_shortly_after_dict = {}
        for n in notes_shortly_after:
            if n['note_on'] not in notes_shortly_after_dict:
                notes_shortly_after_dict[n['note_on']] = []
            notes_shortly_after_dict[n['note_on']].append(n)
            
        drag_time = np.random.uniform(0.2, 0.8) * min(note['duration_sec'][0], MAX_DUR4DRAG)
        if not self.change_tracker.change_note_offset(note['onset_sec'], note['pitch'], drag_time, 'drag'):
            print('exit drag function for initial pitch not found')
            return

        drag_time_accum = drag_time
        for key, n_list in notes_shortly_after_dict.items():
            ripple_drag_time_n = drag_time * np.random.random()
            n = n_list[0]
            start_time = n['note_on']
            self.change_tracker.time_offset(start_time, ripple_drag_time_n, 'drag') 
            for n in n_list:
                self.change_tracker.change_note_offset(n['note_on'], n['pitch'], 
                                                   ripple_drag_time_n * np.random.uniform(0.8, 1.2), 'drag')
            drag_time_accum += ripple_drag_time_n

        print(f"added rhythm drag from note {note['id']}.")
