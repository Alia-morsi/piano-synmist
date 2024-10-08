
import partitura as pt
import mido
import math
import numpy as np
import pandas as pd
import numpy.lib.recfunctions as rfn
import copy
from region_classifier import RegionClassifier
import lowlvl 
import csv
from utils import payload_to_csv, timemap_to_csv

#Parametrizations:
#forward_backwards
#forward=(np.random.random() > 0.5??
#This would be implemented as a class that maintains the src(whether score or rendered perf) and tgt,
#and handle the labelling. 

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
    
    payload = sorted(payload, key=lambda x: (x[0], precedence(x[1]))) #first by time, then by operation
    return payload

def print_payload_item(p):
    print('{} - {}'.format(p[0], p[1]))
def print_payload_list(payload):
    for p in payload:
        print_payload_item(p)


class Mistaker():
    def __init__(self, performance_path, time_series_annotation=None):
        """This is the functionality for detecting / parsing regions of interest, 
        based on the assumption that errors are usually associated with certain 
        regions / technique groups and the probability of making mistakes varies 
        depending on the musical context.
        beat annotation to be given as an array time values with their gt label. for now just 1d.
        Args:
            performance_path (str): 
        """
        self.performance_path = performance_path
        #self.performance = pt.load_performance(performance_path)
        self.performance = pt.load_performance(performance_path, pedal_threshold=127)
        self.time_series_annotation = time_series_annotation

        #change tracker must be instantiated before the other fields are added from region classifier
        self.change_tracker = lowlvl.lowlvl(self.performance.note_array(), time_series_annotation)
        
        rc = RegionClassifier(performance_path, save=False)

        self.na = rc.na     # note array with region classification
        self.white_keys, self.black_keys = self.black_white_keys()

        self.sampling_prob = pd.read_csv("sampling_prob.csv", index_col='index')

    def schedule_mistakes(self):
        self.mistake_scheduler()

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
                #is it even possible to have numbers less than 10?
            }):
        """Based on heuristics and classified regions, create mistakes.
            n_mistakes: the number of mistakes to add for each group. 
        """
        payload = []
        for mistake_type, n in n_mistakes.items():
            counter = 0
            # turn the predefined probablity into an array to decide with texture region is chosen.
            #why is p multiplied by 10
            sample_array = [[texture_group] * int(p * n_mistakes[mistake_type]) for texture_group, p in self.sampling_prob[mistake_type].items()]
            sample_array = [y for x in sample_array for y in x]

            #check sample_array
            for i in range(0, n):
                # generate the mistakes around the regions that's probable.
                texture_sampler = np.random.random()
                texture_group = sample_array[int(texture_sampler * len(sample_array))]
                note = self.sample_group(self.na, texture_group)

                if not len(note): # no note in this texture group. Remove this group from the sample_array so don't sample a next time.
                    sample_array = list(filter(lambda a: a != texture_group, sample_array))
                    continue

                #to be filled with tupes as: function name, note, args dict
                rollback_dice = np.random.random()   

                if mistake_type == 'forward_backward_insertion':
                    # TODO: get the ascending parameter.
                    payload.append((note['onset_sec'], 'forward_backward_insertion', {'note': note, 'forward': np.random.random() > 0.5}))
                if mistake_type == 'mistouch':
                    payload.append((note['onset_sec'], 'mistouch', {'note': note,}))
                if mistake_type == 'pitch_change':
                    payload.append((note['onset_sec'], 'pitch_change', {'note': note,}))
                    if rollback_dice < rollback_association_prob['pitch_change']:
                        payload.append((note['onset_sec'], 'rollback', {'note': note, 'events_back_range': (0,5)}))
                if mistake_type == 'drag': 
                    payload.append((note['onset_sec'], 'drag', {'note': note,}))
                    #doesn't make sense to add it after the first note of the drag.
                    #makes sense to add it after the whole drag window, but to return to 
                    #the first note dragged.
                    if rollback_dice < rollback_association_prob['drag']:
                        payload.append((note['onset_sec'], 'rollback', {'note': note, 'events_back_range': (0,5)}))

        #payload_dtype = dtypes = [('start_time', 'U20'), ('data', 'O')]
        #payload = np.array(payload, dtype=), #consider, just to make the next line more readable
        payload = sort_payload(payload)
        
        for i, p in enumerate(payload):
            try:
                self.__getattribute__(p[1])(**p[2])
            except Exception as e:
                    print(e)
        return payload

    def sample_group(self, data, group):
    #either returns a note belonging to a texture group, or 0 if it there aren't notes in this texture
    #group
        mask = data[group] == 1
        group_data = data[mask]
        if len(group_data) == 0:
            print(f"no data in group {group}.")
            return group_data
        return group_data[np.random.choice(len(group_data), 1, replace=True)]

    ########### Mid Level Mistake Functions ############
    def rollback(self, note, events_back_range):
        #what note am i passing in is the note I want to go back from.
        #TODO: should have something to permit changes to the notes depending on 'something'.. 
        num_events_back = np.random.randint(events_back_range[0], events_back_range[1])
        idx, notes_to_repeat = self.change_tracker.get_notes(note['onset_sec'], num_events_back)
        #idx probably won't be used at all.
        #make these notes start from 0, not from whatever their start point was
        #modify the values of its parameters if needed. TBD for later

        onset_shift = notes_to_repeat['onset_sec'][0] #the first onset in the notes to be repeated
        notes_to_repeat['onset_sec'][:] -= onset_shift

        #I wonder if I need to add some little buffer of silence.
        #I add the duration to note['onset_sec'] because I want the addition to happen after the note
        #ends.

        #hesitation before repeating notes
        hesitation = 0.2 #randomize
        self.change_tracker.time_offset(note['onset_sec']+ note['duration_sec'], hesitation, 'rollback')
        self.change_tracker.go_back(onset_shift, note['onset_sec']+ note['duration_sec'], notes_to_repeat)
        return

    def forward_backward_insertion(self, note, forward=True, ascending=True):
        """insert notes that belongs to the previous / later onset
        note: The note that we would like to have mistake inserted around. 
        forward: look for future or past neighbor to double.
        ascending: when having multiple candidates, whether to insert note with higher pitch. 
        marking: whether to mark the inserted error with a bottom note.
        """
        #Choice of note.
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

        #maybe remove.. let's see. function could just exit if no appropriate one found.
        if not len(insert_pitches_):
            insert_pitches_ = insert_pitches
        insert_pitch = np.random.choice(insert_pitches_)
        
        #find nn in this context
        #are these to make it sound human, or are these to add mistakes? 
        #perhaps it is better that 'humanizing the score is left to rendering, to not mix many thigns'
        #anyway, if the wiggling is for it to be a mistake (since it's rendered), 
        #then we should implemented as a function in lowlvl, so that a label is made
        onset = note['onset_sec'][0] + np.random.uniform(low=0.0, high=0.5) * 0.05 # 50ms of wiggle space in onset
        duration = note['duration_sec'][0] + np.random.uniform(low=0.0, high=0.5) * 0.05 
        velocity = int(((np.random.random() * 0.5) + 0.5) * note['velocity'])

        self.change_tracker.pitch_insert(onset, insert_pitch['pitch'], duration, velocity, "fwdbackwd") 
        print(f"added forward={forward} insertion at note {note['id']} with pitch {insert_pitch['pitch']}.")


    def mistouch(self, note):
        """add mistouched inserted note for the given note."""
        #the insertion here should be much shorter than the forward backward.
        # white keys can't touch black keys
        if note['pitch'] in self.white_keys: 
            insert_pitch = self.white_keys[self.white_keys.index(note['pitch']) + (np.random.choice([1, -1]))]
            assert(insert_pitch != note['pitch'])
        else:
            insert_pitch = note['pitch'] + (np.random.choice([1, -1]))

        #TODO: Calculate duration and velocity in a sensible way
        duration = 0.2
        velocity = 60

        self.change_tracker.pitch_insert(note['onset_sec'], insert_pitch, duration, velocity, "mistouch")
        print(f"added mistouch insertion at note {note['id']} with pitch {insert_pitch}.")

    #probably pitch change the same as the confident substitution. 
    #wonder if the texture should affect which pitch would be mistaken..
    def pitch_change(self, note, rollback=False, change_chordblock=False):
        """change the pitch of the given note.
        the changed pitch can be either: the pitch-wise nearest neighboring note, or random 1/2 semitones around
        (TODO)change_chordblock: if True, all notes within that block would change into adjacent chord. 
        (TODO)rollback: if rollback, the error is combined with a drag and and then correct pitch
        """
        #get the note info
        #delete it
        #place another with a changed pitch (which pitch?)
        #in v1 the implementation was that it is closer to a neighbour note. that one I changed.

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

    #Make the rhythm mistakes for tomorrow..
    ########### Functions for rhythm mistakes ####################
    #range of dragtime is potentially parametrizable.
    def drag(self, note, drag_window=5):
        """a drag / hesitation on the note position 
        details: 1. after the drag, restarted passages will be slower
            2. the restarted passages also come with lower volume. 
            drag window the window after the starting note for which drags will continue happening.
        """
        # dragging can happen at most 1 - 3 times the duration of the note
        
        #logic:
        #mark the note at the given id.
        #then, if the note being processed is close to the onset of the input note by 0.05
        #then drag the offset
        #and, shift notes.
        #import pdb
        #pdb.set_trace()

        #if within drag window:
        #we hardcode 1.0 as the maximum duration of any note.
        notes_shortly_after = [n for n in self.performance.performedparts[0].notes 
                               if 0 < n['note_on'] - note['onset_sec'] <= min(note['duration_sec'], MAX_DUR4DRAG) * drag_window]

        #convert it to a tuple of lists based on onsets
        notes_shortly_after_dict = {}
        for n in notes_shortly_after:
            if n['note_on'] not in notes_shortly_after_dict:
                notes_shortly_after_dict[n['note_on']] = []
            notes_shortly_after_dict[n['note_on']].append(n)
            
        # drag the offset of the identified notes 
        # identify a way to change the velocity until 'normal' playing is resumed.. (gradual decrease? etc)
        # for every note drag, timeshift the notes that come after
        drag_time = np.random.uniform(0.2, 0.8) * min(note['duration_sec'][0], MAX_DUR4DRAG) #otherwise everything becomes an array..
        if not self.change_tracker.change_note_offset(note['onset_sec'], note['pitch'], drag_time, 'drag'):
            print('exit drag function for initial pitch not found')
            #mport pdb
            #pdb.set_trace()
            return

        #open quesiton: do drag times start shorter and then get longer, or the other way round??
        #the mult by 1.5 is to test how it sounds when we assume it will get longer. 
        #maybe that should also be something randomized. 

        #another drag artifact is that probably if there are multiple notes 
        #at the same time, their drag should not be added to the accum function!..
        #this explains why we observe longer offsets after a drag is a applied at the site of a chord
        #Although it's a bug, it is sensible behaviour because it would take more human processing
        #time to reflect on all the notes pressed and identify the source of the problem.

        #the accumulator logic would have only made sense if we changed the notes offset only in the loop,
        #accumulated the total offset, and then did the shift after the loop. This approach would allow
        #us to handle the case highlighted above, where offsets applied to notes which are played at the
        #same time should not accumulate.
        #removed the mult of drag_time * 1.5 for ripple drag time.
        drag_time_accum = drag_time
        for key, n_list in notes_shortly_after_dict.items():
            ripple_drag_time_n =  drag_time * np.random.random()
            n = n_list[0]
            start_time = n['note_on']
            self.change_tracker.time_offset(start_time, ripple_drag_time_n, 'drag') 
            for n in n_list:
                #this assumes that both the notes will drag with the same time
                self.change_tracker.change_note_offset(n['note_on'], n['pitch'], 
                                                   ripple_drag_time_n * np.random.uniform(0.8, 1.2), 'drag')
                
            drag_time_accum += ripple_drag_time_n

            #add the wiggle velocity function

            # shift forward all later notes by the drag time 
            #if (nn['note_on'] - note['onset_sec']) > 0.05:
                # for the neighboring notes, slow down their tempo and decrease volume
            #    
            #        nn['note_on'] += drag_time * ((np.random.random() * 0.5) + 0.5) 
            #        nn['note_off'] += drag_time * ((np.random.random() * 0.5) + 0.5) 
            #        nn['velocity'] = int(((np.random.random() * 0.3) + 0.7) * nn['velocity'])
           
        print(f"added rhythm drag from note {note['id']}.")


#when this becomes a package we will have to remove main so that the package is not dependent on
#any specific dataset adapter
if __name__ == '__main__':
    import os
    import pathlib
    import argparse
    import tqdm
    import sys
    from labeled_dataset_adapters.asap_adapter import ASAPLoader

    parser = argparse.ArgumentParser(description='System to overlay synthetic mistakes on mistake free midi piano performances, whether rendered or real ones')
    parser.add_argument("in_folder", type=pathlib.Path, help="root folder with midi files to be modified with errors")
    parser.add_argument("out_folder", type=pathlib.Path, help="root output folder")
    parser.add_argument("run_id", help="identifier for this run. used to name out folder")
    parser.add_argument("--ts_annot", action='store_true', help="maintain changes to timeseries annotations")
    parser.add_argument('--no_ts_annot', action='store_false', dest='ts_annot')
    p = parser.parse_args()

    if not p.in_folder.exists():
        print('in_folder does not exist')
        sys.exit()

    pathlib.Path(os.path.join(p.out_folder, p.run_id)).mkdir(parents=True, exist_ok=True)
    
    #pathlib.Path(os.path.join(p.run_id, p.out_folder)).mkdir(parents=True, exist_ok=False)

    #makes no sense to make it super general since at the end there is dataset specific stuff 
    #handle this before making a package..

    if p.ts_annot:
        filelist = ASAPLoader(p.in_folder)
    else:
        filelist = [zip(i , []) for i in p.in_folder.glob('*/*.mid')] #this will ofc be specific to the given data folder

    for (fn_mid, ts_annot) in tqdm.tqdm(filelist): #maybe this will crash if it is from the non ASAPLoader path.
        try:
            mk = Mistaker(fn_mid, ts_annot)
            payload = mk.mistake_scheduler()

            stem = os.path.split(os.path.splitext(fn_mid)[0])[1]

            #calculate specific out folder if needed.. ideally should have a flag for recursive or not

            if p.ts_annot:
                relative_portion = os.path.relpath(os.path.dirname(fn_mid), p.in_folder)
                out_folder = os.path.join(p.out_folder, p.run_id, relative_portion, )
                pathlib.Path(out_folder).mkdir(parents=True, exist_ok=True)

            else:
                out_folder = os.path.join(p.out_folder, p.run_id)

            #add ground truth boolean, and add the annotation file example. use ASAP for now since I have that already
            mk.change_tracker.get_target_miditrack(os.path.join(out_folder, '{}-tgt.mid'.format(stem)))
            mk.change_tracker.get_src_miditrack(os.path.join(out_folder, '{}-src.mid'.format(stem)))
            mk.change_tracker.get_label_miditrack(os.path.join(out_folder, '{}-mistake-label.mid'.format(stem)))
            if p.ts_annot:
                adj_gt = mk.change_tracker.get_adjusted_gt()
                src_gt = mk.change_tracker.ts_annot
                np.savetxt(os.path.join(out_folder,'{}tgt-gt-label.csv'.format(stem)), adj_gt,  delimiter = ",")
                np.savetxt(os.path.join(out_folder,'{}src-gt-label.csv'.format(stem)), src_gt,  delimiter = ",")

            payload_to_csv(payload, os.path.join(out_folder, '{}-label.csv'.format(stem)))            
            timemap_to_csv(mk.change_tracker.get_timemap(), mk.change_tracker.repeat_tracker, os.path.join(out_folder, '{}-mistake_timemap.csv'.format(stem)))

        except OSError:
            pass

