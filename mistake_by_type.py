
import partitura as pt
import numpy as np
import pandas as pd
import numpy.lib.recfunctions as rfn
import copy
from region_classifier import RegionClassifier
import hook


class Mistaker():
    def __init__(self, performance_path, burgmuller=False):
        """This is the functionality for detecting / parsing regions of interest, 
        based on the assumption that errors are usually associated with certain 
        regions / technique groups and the probability of making mistakes varies 
        depending on the musical context.

        Args:
            performance_path (str): 
        """

        self.performance_path = performance_path
        self.performance = pt.load_performance(performance_path)

        rc = RegionClassifier(performance_path, save=False)

        self.na = rc.na     # note array with region classification
        self.white_keys, self.black_keys = self.black_white_keys()

        self.sampling_prob = pd.read_csv("sampling_prob.csv", index_col='index')
        self.mistake_scheduler()

        # self.forward_backward_insertion(self.na[self.na['id'] == 'n17'])

        pt.save_performance_midi(self.performance, performance_path[:-4] + "_mk.mid")


    def black_white_keys(self):
        white_keys_all_octaves = []
        black_keys_all_octaves = []

        for octave in range(0, 11):  # MIDI octaves range from 0 to 10
            # The MIDI number for C0 is 12. Each octave adds 12 to this base.
            base_midi_number = 12 + octave * 12

            # White keys are all the natural notes (C, D, E, F, G, A, B)
            white_keys = [base_midi_number, base_midi_number + 2, base_midi_number + 4,
                        base_midi_number + 5, base_midi_number + 7, base_midi_number + 9,
                        base_midi_number + 11]

            # Black keys are all the sharps/flats
            black_keys = [base_midi_number + 1, base_midi_number + 3, base_midi_number + 6,
                        base_midi_number + 8, base_midi_number + 10]

            white_keys_all_octaves.extend(white_keys)
            black_keys_all_octaves.extend(black_keys)

        return white_keys_all_octaves, black_keys_all_octaves


    ########### Function for scheduling mistakes #################
    def mistake_scheduler(self, n_mistakes={
                "forward_backward_insertion": 10,
                "mistouch": 10,
                "pitch_change": 10,
                "drag": 10
            }):
        """Based on heuristics and classified regions, add mistake on the  
        
            n_mistakes: the number of mistakes to add for each group. 
        """

        # for each type of mistake, we add 
        for mistake_type, n in n_mistakes.items():
            counter = 0
            # turn the predefined probablity into an array to decide with texture region is chosen.
            sample_array = [[texture_group] * int(p * 10) for texture_group, p in self.sampling_prob[mistake_type].items()]
            sample_array = [y for x in sample_array for y in x] 

            while n:
                # generate the mistakes around the regions that's probable.
                prob = np.random.random()
                texture_group = sample_array[int(prob * len(sample_array))]
                note = self.sample_group(self.na, texture_group)
                # change: sort the mistake decisions and then apply them sequentially.

                if not len(note): # no note in this texture group. Remove this group from the sample_array so don't sample a next time.
                    sample_array = list(filter(lambda a: a != texture_group, sample_array))
                    continue

                if mistake_type == 'forward_backward_insertion':
                    # TODO: get the ascending parameter.
                    try:
                        self.forward_backward_insertion(note, forward=(np.random.random() > 0.5), marking=True)
                    except Exception as e:
                        print(e)
                if mistake_type == 'mistouch':
                    self.mistouch(note, marking=True)
                if mistake_type == 'pitch_change':
                    self.pitch_change(note, marking=True)
                if mistake_type == 'drag':
                    self.drag(note, marking=True)

        return

    def sample_group(self, data, group):
        mask = data[group] == 1
        group_data = data[mask]
        if len(group_data) == 0:
            print(f"no data in group {group}.")
            return group_data
        return group_data[np.random.choice(len(group_data), 1, replace=True)]


    ########### Functions for adding various mistakes ############

    def forward_backward_insertion(self, note, forward=True, ascending=True, marking=False):
        """insert notes that belongs to the previous / later onset
    
        note: The note that we would like to have mistake inserted around. 
        forward: look for future or past neighbor to double.
        ascending: when having multiple candidates, whether to insert note with higher pitch. 
        marking: whether to mark the inserted error with a bottom note.
        """

        # forward: search for the nearest future (forward) neighbor and double it
        if forward:
            insert_pitches = self.na[self.na['onset_sec'] > note['offset_sec']]
            min_onset = insert_pitches['onset_sec'].min()
            insert_pitches = insert_pitches[np.isclose(insert_pitches['onset_sec'], min_onset, atol=0.05)]
        else: # backward (past neighbor)
            insert_pitches = self.na[self.na['offset_sec'] < note['onset_sec']]
            max_onset = insert_pitches['onset_sec'].max()
            insert_pitches = insert_pitches[np.isclose(insert_pitches['onset_sec'], max_onset, atol=0.05)]

        if (ascending and forward) or ((not ascending) and (not forward)): # insert a pitch that's higher
            insert_pitches_ = insert_pitches[insert_pitches['pitch'] > note['pitch']]
        else:
            insert_pitches_ = insert_pitches[insert_pitches['pitch'] < note['pitch']]

        if not len(insert_pitches_):
            insert_pitches_ = insert_pitches
        insert_pitch = np.random.choice(insert_pitches_)

        notes_cpy = copy.deepcopy(self.performance.performedparts[0].notes)
        for nn in self.performance.performedparts[0].notes:
            if nn['id'] == note['id']:
                nn_cpy = copy.deepcopy(nn)
                nn_cpy['midi_pitch'] = int(insert_pitch['pitch'])
                nn_cpy['note_on'] = nn['note_on'] + (np.random.random() - 0.5) * 0.05 # 50ms of wiggle space in onset
                nn_cpy['note_off'] = nn['note_off'] + (np.random.random() - 0.5) * 0.05 
                nn_cpy['velocity'] = int(((np.random.random() * 0.5) + 0.5) * nn['velocity']) # randomly decrease the volume of the inserted note (but not less than half).
                # TODO: fix the id of the copied note

                notes_cpy.append(nn_cpy)
                if marking: # change the logic of annoation
                    marking_note = copy.deepcopy(nn_cpy)
                    marking_note['midi_pitch'] = 3
                    marking_note['velocity'] = 120
                    notes_cpy.append(marking_note)
        
        self.performance.performedparts[0].notes = notes_cpy
        print(f"added forward={forward} insertion at note {note['id']} with pitch {insert_pitch['pitch']}.")


    def mistouch(self, note, marking=False):
        """add mistouched inserted note for the given note."""

        # white keys can't touch black keys
        if note['pitch'] in self.white_keys: 
            insert_pitch = self.white_keys[self.white_keys.index(note['pitch']) + (np.random.choice([1, -1]))]
            assert(insert_pitch != note['pitch'])
        else:
            insert_pitch = note['pitch'] + (np.random.choice([1, -1]))

        notes_cpy = copy.deepcopy(self.performance.performedparts[0].notes)
        for nn in self.performance.performedparts[0].notes:
            if nn['id'] == note['id']:
                nn_cpy = copy.deepcopy(nn)
                nn_cpy['midi_pitch'] = int(insert_pitch)
                nn_cpy['note_on'] = nn['note_on'] + (np.random.random() - 0.5) * 0.05 # 50ms of wiggle space in onset
                nn_cpy['note_off'] = nn['note_off'] + (np.random.random() - 0.5) * 0.05 
                nn_cpy['velocity'] = int(((np.random.random() * 0.5) + 0.5) * nn['velocity']) # randomly decrease the volume of the inserted note (but not less than half).

                notes_cpy.append(nn_cpy)
                if marking:
                    marking_note = copy.deepcopy(nn_cpy)
                    marking_note['midi_pitch'] = 3
                    marking_note['velocity'] = 100
                    notes_cpy.append(marking_note)
        
        self.performance.performedparts[0].notes = notes_cpy
        print(f"added mistouch insertion at note {note['id']} with pitch {insert_pitch}.")

    

    def pitch_change(self, note, rollback=False, change_chordblock=False, marking=False):
        """change the pitch of the given note.

        the changed pitch can be either: the pitch-wise nearest neighboring note, or random 1/2 semitones around
        (TODO)change_chordblock: if True, all notes within that block would change into adjacent chord. 
        (TODO)rollback: if rollback, the error is combined with a drag and and then correct pitch
        """
        notes_cpy = copy.deepcopy(self.performance.performedparts[0].notes)
        for nn in notes_cpy:
            if nn['id'] == note['id'] and nn['midi_pitch'] == note['pitch']:
                marking_note = copy.deepcopy(nn) 
                changed_pitch = nn['midi_pitch']
                if np.random.random() > 0.5:
                    neighbor_pitches = self.na[self.na['offset_sec'] < note['onset_sec']]
                    if len(neighbor_pitches):
                        max_onset = neighbor_pitches['onset_sec'].max()
                        neighbor_pitches = neighbor_pitches[np.isclose(neighbor_pitches['onset_sec'], max_onset, atol=0.05)]
                        near_neighbor = neighbor_pitches[np.abs(neighbor_pitches['pitch'] - note['pitch']).argmin()]
                        changed_pitch = near_neighbor['pitch']
                else:
                    changed_pitch = nn['midi_pitch'] + np.random.choice([-2, -1, 1, 2])
                nn['midi_pitch'] = int(changed_pitch)


                if marking:
                    marking_note['midi_pitch'] = 3
                    marking_note['velocity'] = 90
                    notes_cpy.append(marking_note)

                self.performance.performedparts[0].notes = notes_cpy
                print(f"added pitch change at note {note['id']} with pitch {changed_pitch}.")


    ########### Functions for rhythm mistakes ####################


    def drag(self, note, marking=False):
        """a drag / hesitation on the note position 
        
        details: 1. after the drag, restarted passages will be slower
            2. the restarted passages also come with lower volume. 
        
        """

        # dragging can happen at most 1 - 3 times the duration of the note
        drag_time = ((np.random.random() * 2) + 1) * note['duration_sec']

        for nn in self.performance.performedparts[0].notes:
            if nn['id'] == note['id']:
                marking_note = copy.deepcopy(nn)

            # drag the offset of given note. 
            if np.isclose(nn['note_on'], note['onset_sec'], atol=0.05):
                nn['note_off'] += drag_time * np.random.random()

            # shift forward all later notes by the drag time 
            if (nn['note_on'] - note['onset_sec']) > 0.05:
                # for the neighboring notes, slow down their tempo and decrease volume
                if (nn['note_on'] - note['onset_sec']) < note['duration_sec'] * 3:
                    nn['note_on'] += drag_time * ((np.random.random() * 0.5) + 0.5) 
                    nn['note_off'] += drag_time * ((np.random.random() * 0.5) + 0.5) 
                    nn['velocity'] = int(((np.random.random() * 0.3) + 0.7) * nn['velocity'])
                else:
                    nn['note_on'] += drag_time
                    nn['note_off'] += drag_time

        notes_cpy = copy.deepcopy(self.performance.performedparts[0].notes)
        if marking:
            marking_note['midi_pitch'] = 3
            marking_note['velocity'] = 1
            notes_cpy.append(marking_note)

        self.performance.performedparts[0].notes = notes_cpy
        print(f"added rhythm drag at note {note['id']}.")


if __name__ == '__main__':


    import glob
    for path in glob.glob("/Users/huanzhang/01Acdemics/PhD/Research/SynthMistakes/data_processing/burgmuller/*[!k].mid"):
        mk = Mistaker(path)

    # mk = Mistaker("/Users/huanzhang/01Acdemics/PhD/Research/Datasets/Burgmuller/b-02-annot.mid", burgmuller=True)