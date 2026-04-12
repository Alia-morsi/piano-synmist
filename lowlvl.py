import partitura as pt
import mido
import math
import numpy as np
import copy
import pretty_midi
import os
import pdb

#MIDI Locations for the labels so that we can decipher what's being output.

MID_FWDBCKWD = 21      #A0
MID_ROLLBACK = 25      #C#1
MID_MISTOUCH = 28      #E1
MID_WRNG_PRED_INS = 30 #F#1
MID_DRAG = 33          #A1


LOW_INSERT = 1 #C#-1
LOW_DELETE = 3 #D#-1
LOW_GOBACK = 5 #F-1
LOW_SHIFT = 7  #G-1
LOW_CHG_OFFSET = 9 #A-1
LOW_CHG_ONSET = 11 #B-1

DEFAULT_MID = 20 #G#0
DEFAULT_LOW = 0  #C-1

mid_lvl_opers = {
    'MID_FWDBCKWD': MID_FWDBCKWD,
    'MID_ROLLBACK': MID_ROLLBACK,
    'MID_MISTOUCH': MID_MISTOUCH,
    'MID_WRNG_PRED_INS': MID_WRNG_PRED_INS,
    'MID_DRAG': MID_DRAG,
    'DEFAULT_MID': DEFAULT_MID,
}

low_lvl_opers = {
    'LOW_INSERT': LOW_INSERT,
    'LOW_DELETE': LOW_DELETE,
    'LOW_GOBACK': LOW_GOBACK,
    'LOW_SHIFT': LOW_SHIFT,
    'LOW_CHG_OFFSET': LOW_CHG_OFFSET,
    'LOW_CHG_ONSET': LOW_CHG_ONSET,
    'DEFAULT_LOW': DEFAULT_LOW,
}

mid_label_pitch_map = {
    'fwdbackwd' : MID_FWDBCKWD, 
    'rollback': MID_ROLLBACK, 
    'wrong_pred': MID_WRNG_PRED_INS, 
    'mistouch': MID_MISTOUCH, 
    'drag': MID_DRAG
}
low_label_pitch_map = {
    'pitch_insert': LOW_INSERT,
    'pitch_delete': LOW_DELETE, 
    'time_shift': LOW_SHIFT, 
    'go_back': LOW_GOBACK, 
    'change_offset': LOW_CHG_OFFSET, 
    'change_onset': LOW_CHG_ONSET
}

LABEL_VELOCITY = 10

label_na_fields = [('onset_sec', '<f4'), ('duration_sec', '<f4'), ('pitch', '<i4'), ('velocity', '<i4'), 
              ('midlvl_label', '<U256'), ('lowlvl_label', '<U256')]
regular_na_fields = [('onset_sec', '<f4'), ('duration_sec', '<f4'), ('onset_tick', '<i4'), ('duration_tick', '<i4'), 
                     ('pitch', '<i4'), ('velocity', '<i4'), ('track', '<i4'), ('channel', '<i4'), 
                     ('id', '<U256')]

#contemplate the order of applying the mistakes and if sorting is needed:
#time based
#then, if several at the same time: insert, delete, goback, offset.
#what happens if we delete after an insertion, could we end up deleting what we just inserted?

#TODO: have a function that merges mid_level labels done by the same 'event'. 
# group_mid (start, end), and would merge the notes of the same mid level event into 1.

#Most Likely, ts_annot will not be supported in segmented practice
class lowlvl:
    def __init__(self, src_na, mode='runthrough', ts_annot=[]):
        #the labels would be maintained wrt to timeto or the tgt_na 
        #target = using the nearest giventime, get the timeto. In thoery, all existing score notes will
        #have a timefrom value in the safe zones.
        #ts_annot = time series annotation
        #TODO: if src is an empty note array, fail and exit

        self.mode = mode

        start = src_na['onset_sec'][0]
        end = src_na['onset_sec'][-1]+ src_na['onset_sec'][-1]
        self.time_from = np.linspace(start, end, int(math.ceil(end)-math.ceil(start))*40) #not sure of resolution..
        if self.mode == 'runthrough':
            self.time_to = self.time_from.copy()
        elif self.mode == 'segmented':
            self.time_to = np.full_like(self.time_from, -1) 

        self.repeat_tracker = {} #dictionary to hold the time_from -> time_to mappings for performance regions that will be removed
                                 #the format is: (interval for tgt time region) -> ([target time points], [src time points])
        self.ts_annot = ts_annot

        self.onsets = src_na['onset_sec']
        self.src_na = src_na
        if self.mode == 'runthrough':
            self.tgt_na = copy.deepcopy(self.src_na)
        elif self.mode == 'segmented':
            self.tgt_na = np.zeros(0, dtype=regular_na_fields)
 
        self.label_na = np.zeros(0, dtype=label_na_fields)

        #self.time_res = 0.05 #mostly used for the time_offset calculation.. let's see.
        return

    def _apply_warping_path_offsets(self, tgt_time_to_apply_offset, offset):
        # Note that could change the keys of self.repeat, since the key was the time_to value..

        # function meant to handle all offsets applied to the time_tos, whether the main one, or those stored in repeat_tracker.
        # we need to make sure that if an offset is applied, that all time_tos covering points after also get shifted.
        # find all the points in time_to (whether in repeat_tracker, or in self.time_to, that have
        
        #search self.time_to
        self_time_to_Idxs = np.where(self.time_to >= tgt_time_to_apply_offset)[0] 
        self.time_to[self_time_to_Idxs] += offset

        #then search all self.repeat_tracker time_to
        #might cause a problem in case the key,val iteration is by reference.. if we delete the entry in the array maybe it could crash..
        replacement_entries = {}
        keys_to_delete = []
        for key, val in self.repeat_tracker.items():
            sub_time_to_Idxs = np.where(val[0] >= tgt_time_to_apply_offset)[0]
            if len(sub_time_to_Idxs) == 0:
                continue

            keys_to_delete.append(key)
            val[0][sub_time_to_Idxs] += offset 
            replacement_entries[(val[0][-1], val[0][0])] = val
            
        for key_to_del in keys_to_delete:
            self.repeat_tracker.pop(key_to_del)
        
        for new_key, new_val in replacement_entries.items():
            self.repeat_tracker[new_key] = new_val
        

    def _repeat_tracker_order(self, src_time):
        #how many repeats exist for a src_time, and what is their order.. since the calling function should pass an index along with the src_time..

        #return all the keys for which a given self.time_from[nearestIdx_src_time] exists.
        nearestIdx_src_time = np.fabs(self.time_from - src_time).argmin()
        start_of_timefrom_window = self.time_from[nearestIdx_src_time]
        
        sorted_repeat_tracker = dict(sorted(self.repeat_tracker.items()))

        keys_list = list(sorted_repeat_tracker.keys())
        mapping_list = list(sorted_repeat_tracker.values())

        idxs_to_include = [idx for idx, (timeto_sub, timefrom_sub) in enumerate(mapping_list) if start_of_timefrom_window in timefrom_sub]
       
        ordered_keys_incl_src = [keys_list[i] for i in idxs_to_include]
        return ordered_keys_incl_src, keys_list   #return sorted list of all the keys which include src, and all the sorted keys. 
        #the sorted keys (meaning that they are sorted by tgt_na start time), would help us determine which repeat pairs need temporal adjustments

    def _create_segmented_practice(self, segments, time_gap=3): 
        #time gap is the gap between inserted segments
        if self.mode != 'segmented':
            return

        for start, end in segments: #these start and end times should be relevant to the src signal
            new_notes_na = self.get_notes_between(start, end)
            if len(new_notes_na) == 0:
                continue

            #we take for granted that the tgt_na is sorted
            if len(self.tgt_na) == 0:
                insertion_offset = 0
            else:
                insertion_offset = self.tgt_na[-1]['onset_sec'] + self.tgt_na[-1]['duration_sec'] + time_gap
            #add a copy of this segment in the target na

            new_notes_start_time = new_notes_na[0]['onset_sec'] #might be slightly different than the given range.
            new_notes_end_time = new_notes_na[-1]['onset_sec']+new_notes_na[-1]['duration_sec']

            for new_note in new_notes_na: #truncate their start time
                new_note['onset_sec'] -= new_notes_start_time
                new_note['onset_sec'] += insertion_offset

            self.tgt_na = np.concatenate((self.tgt_na, new_notes_na))
            self.tgt_na.sort(order='onset_sec')

            #what are the time_from indexes?
            nearestIdx_new_notes_start = np.fabs(self.time_from - new_notes_start_time).argmin()
            nearestIdx_new_notes_end = np.fabs(self.time_from - new_notes_end_time).argmin()

            if not np.all(self.time_to[nearestIdx_new_notes_start:nearestIdx_new_notes_end+1] == -1):
                #then we need to save the old repeat in the repeats structure:
                self.repeat_tracker[(self.time_to[nearestIdx_new_notes_start], self.time_to[nearestIdx_new_notes_end])] = (self.time_to[nearestIdx_new_notes_start:nearestIdx_new_notes_end].copy(), self.time_from[nearestIdx_new_notes_start:nearestIdx_new_notes_end].copy())
            
            #Offset time_to so it maps to where the notes actually land in tgt_na.
            #Notes were shifted by (-new_notes_start_time + insertion_offset),
            #so time_to needs the same shift applied to time_from.
            self.time_to[nearestIdx_new_notes_start:nearestIdx_new_notes_end] = self.time_from[nearestIdx_new_notes_start:nearestIdx_new_notes_end].copy()
            self.time_to[nearestIdx_new_notes_start:nearestIdx_new_notes_end] += (insertion_offset - new_notes_start_time)
    
    def _label_note(self, start, end, lowlvl_label, midlvl_label):
        mid_label_pitch = mid_label_pitch_map[midlvl_label] if midlvl_label in mid_label_pitch_map else DEFAULT_MID
        low_label_pitch = low_label_pitch_map[lowlvl_label] if lowlvl_label in low_label_pitch_map else DEFAULT_LOW

        #add 2 notes, one mid and one low.
        new_label = np.array([(start, end-start, mid_label_pitch, LABEL_VELOCITY, midlvl_label, lowlvl_label), 
                              (start, end-start, low_label_pitch, LABEL_VELOCITY, midlvl_label, lowlvl_label)], dtype=self.label_na.dtype)
        self.label_na = np.concatenate((new_label, self.label_na))
        self.label_na.sort(order='onset_sec')
        return 
    
    def _shift_labels(self, src_time, offset, repeat_index=0):
        #shift all labels after time 'time' by offset s
        #repeat_index is to find which instance of src_time is the one in question. the default is to just modify the one in time_to
        #it seems a bit counter intuitive to do this based on src_time and to keep searching.. but I don't want to change the api at this moment.

        #only used in case we relax the requirement that changes have to be done from earliest to latest.
        #offset can be +ve or -ve.

        #find the correct tgt time given the src_time and repeat_index:
        #then, shift the labels by modifying the time in tgt_na.

        if len(self.label_na) == 0: #if no labels yet, then nothing to shift..
            return

        ordered_keys_incl_src, ordered_keys_list = self._repeat_tracker_order(src_time)

        #perhaps it is useful to relax the requirement of insertions made in temporal order in runthrough mode
        if repeat_index == 0 or len(ordered_keys_incl_src) == 0:
            nearestIdx = np.fabs(self.time_from - src_time).argmin()
            time_in_tgtna = self.time_to[nearestIdx]
        else:
           (time_to_subarray, time_from_subarray) =  self.repeat_tracker[ordered_keys_incl_src[repeat_index - 1]]
           nearestIdx = np.fabs(time_from_subarray - src_time).argmin()
           time_in_tgtna = time_to_subarray[nearestIdx]

        if time_in_tgtna == -1:
            return

        mask = self.label_na['onset_sec'] >= time_in_tgtna
        self.label_na['onset_sec'][mask] += offset

        return
    
    #this in itself does not shift consequent pitches
    def pitch_insert(self, src_time, pitch, duration, velocity, midlvl_label, repeat_index=0):
        #TODO: find a calculate ticks option from partitura.. 
        #TODO: from the match file, find what id is placed for notes that are 'extra'

        ordered_keys_incl_src, ordered_keys_list = self._repeat_tracker_order(src_time)

        #perhaps it is useful to relax the requirement of insertions made in temporal order in runthrough mode
        if repeat_index == 0 or len(ordered_keys_incl_src) == 0:
            nearestIdx = np.fabs(self.time_from - src_time).argmin()
            tgt_insertion_time = self.time_to[nearestIdx]
        else:
           (time_to_subarray, time_from_subarray) =  self.repeat_tracker[ordered_keys_incl_src[repeat_index - 1]]
           nearestIdx = np.fabs(time_from_subarray - src_time).argmin()
           tgt_insertion_time = time_to_subarray[nearestIdx]

        if tgt_insertion_time == -1:
            print('pitch_insert: src_time {:.3f} maps to unmapped region (time_to=-1), skipping'.format(src_time))
            return
        #instead of using 0, we should convert the seconds time to tick time and initialize this properly....
        new_note = np.array([(tgt_insertion_time, duration, 0, 0, pitch, velocity, 0, 0, 'none')], dtype=self.tgt_na.dtype)
        self.tgt_na = np.concatenate((self.tgt_na, new_note))

        self.tgt_na.sort(order='onset_sec')
        self._label_note(tgt_insertion_time, tgt_insertion_time+duration, 'pitch_insert', midlvl_label)
        return
    
    def _find_note_in_tgt(self, src_time, pitch, repeat_index=0):

        ordered_keys_incl_src, ordered_keys_list = self._repeat_tracker_order(src_time)

        if repeat_index == 0 or len(ordered_keys_incl_src) == 0:
            nearestIdx = np.fabs(self.time_from - src_time).argmin()
            time_in_tgtna = self.time_to[nearestIdx]
        else:
           (time_to_subarray, time_from_subarray) =  self.repeat_tracker[ordered_keys_incl_src[repeat_index - 1]]
           nearestIdx = np.fabs(time_from_subarray - src_time).argmin()
           time_in_tgtna = time_to_subarray[nearestIdx]

        if time_in_tgtna == -1:
            print('_find_note_in_tgt: src_time {:.3f} maps to unmapped region (time_to=-1)'.format(src_time))
            return False, None, None

        window = 0.05 #a window 50 ms before and after for trying to find the onset of the pitch in question.

        #we could evade little time offset problems by:
        #1. setting a tol. window around the src_time (which we do) which we check for all notes of the specified
        # pitch and get the nearest one to the given time.
        #2. find all notes of that pitch in the score and just choose the nearest one within a thresh.
        lowerbound_in_tgt_na = np.fabs(np.array(self.tgt_na['onset_sec'] - (time_in_tgtna - window))).argmin()

        #upperbound isn't done by a oneliner because we need to return the last min. index, not the first:
        tgt_onset_dist_from_upperbound_time = np.fabs(np.array(self.tgt_na['onset_sec']  - (time_in_tgtna + window)))
        upperbound_in_tgt_na = np.where(
            tgt_onset_dist_from_upperbound_time == tgt_onset_dist_from_upperbound_time.min())[0][-1]
        #this indexing above should always have a value if the inputs are not empty..

        found = False
        note_options = []

        #the problem with the lowerbound and upperbound logic is that, in case of chords, there can be several
        #notes at that time, and argmin will only return the index of the first..

        for i in range(lowerbound_in_tgt_na, upperbound_in_tgt_na+1):
            #find all notes with the corresponding pitch
            if self.tgt_na['pitch'][i] == pitch:
                note_options.append((self.tgt_na['onset_sec'][i], i))
                found = True
                
        if found==False:                   
            print('pitch {} not found at time {}'.format(pitch, src_time)) 
            return False, None, None
        else:
            #find closest to the src_time from the list. should be at the top
            note_start, note_idx = sorted(note_options, key=lambda x: np.fabs(x[0] - src_time))[0]

        return found, note_idx, note_start

    
    def pitch_delete(self, src_time, pitch, midlvl_label, repeat_index=0):
        found, note_idx, note_start = self._find_note_in_tgt(src_time, pitch, repeat_index)

        if not found:
            return
        else:
            note_end = note_start + self.tgt_na['duration_sec'][note_idx]
            self.tgt_na = self.tgt_na[~((self.tgt_na['onset_sec'] == self.tgt_na['onset_sec'][note_idx]) & 
                                        (self.tgt_na['pitch'] == self.tgt_na['pitch'][note_idx]))]

            self._label_note(note_start, note_end, 'pitch_delete', midlvl_label)          
        return
    
    #works for shorten note and extend note
    def change_note_offset(self, src_time, pitch, offset_shift, midlvl_label, repeat_index=0):
        found, note_idx, note_start = self._find_note_in_tgt(src_time, pitch, repeat_index)
        #it should always be found.. unless the note has been deleted in prior processing. 
        if not found:
            return False
        self.tgt_na['duration_sec'][note_idx] += offset_shift
        note_end = note_start + self.tgt_na['duration_sec'][note_idx]
        self._label_note(note_start, note_end, "change_offset", midlvl_label)
        return True
    
    def change_note_onset(self, src_time, pitch, onset_shift, midlvl_label, lowlvl_label):
        #TODO: this would have a limit, because if it goes tooooo far, then it should be a deleted note in
        #this region, and a note insertion in the other..
        #this is for wiggles that make sense, without the need to change time_from or time_to.
        return
    
    def _construct_note_na(self, notes):
        #just converts a list of notes into the expected na dtype, to avoid confusions on accessing elements..
        #no sorting is applied.
        #lol this function was to solve a misunderstood problem..
        notes_na = np.zeros(0, dtype=self.src_na.dtype)
        notes_na = np.concatenate((np.array(notes, dtype=self.src_na.dtype), notes_na))
        notes_na.sort(order='onset_sec') 
        return notes_na

    #def get_notes_between(self, src_time_start, src_time_end):
    #    src_onsets = self.src_na['onset_sec']

    #    src_starting_note_idx = np.fabs(src_onsets - src_time_start).argmin() #by default this returns the first occurrence of this value.
    #    #for the end time, to ensure that we get the last element that has the 'end' time:
    #    nearest_value_to_src_time_end = src_onsets[np.fabs(src_onsets - src_time_end).argmin()]
    #    src_ending_note_idx = np.argwhere(np.fabs(src_onsets == nearest_value_to_src_time_end)).flatten().tolist()[-1]
        
    #    new_notes = [copy.deepcopy(note) for note in self.src_na[src_starting_note_idx:src_ending_note_idx+1]]
        
    #    return self._construct_note_na(new_notes)

    def get_notes_between(self, src_time_start, src_time_end):
        src_onsets = self.src_na['onset_sec']

        # First note at or after src_time_start
        src_starting_note_idx = np.searchsorted(src_onsets, src_time_start, side='left')

        # Last note at or before src_time_end
        src_ending_note_idx = np.searchsorted(src_onsets, src_time_end, side='right') - 1

        if src_starting_note_idx > src_ending_note_idx or src_starting_note_idx >= len(src_onsets):
            return self._construct_note_na([])

        new_notes = [copy.deepcopy(note) for note in self.src_na[src_starting_note_idx:src_ending_note_idx+1]]

        return self._construct_note_na(new_notes)

    def get_notes(self, src_time, num_events_back):
        CHORD_TOLERANCE = 0.030  # 30ms, tweak to taste
        #from src_time, go backwards for num_events_back (a chord is considered as one score event, for ex.)
        #if there are not enough backwards notes, stop.
        #num events back 0 is a repeat of the current note.
        src_onsets = self.src_na['onset_sec']
        nearest_src_idx = np.fabs(src_onsets - src_time).argmin()

        back_src_idx = nearest_src_idx
        curr_event_time = self.src_na['onset_sec'][back_src_idx]
        notes = [copy.deepcopy(self.src_na[back_src_idx])]

        # First, collect all notes at the current event time
        while back_src_idx > 0 and abs(self.src_na['onset_sec'][back_src_idx - 1] - curr_event_time) <= CHORD_TOLERANCE:
            notes.append(copy.deepcopy(self.src_na[back_src_idx - 1]))
            back_src_idx -= 1

        while num_events_back:
            if back_src_idx == 0:
                return back_src_idx, self._construct_note_na(notes)

            # Move to the previous event
            num_events_back -= 1
            curr_event_time = self.src_na['onset_sec'][back_src_idx - 1]
            notes.append(copy.deepcopy(self.src_na[back_src_idx - 1]))
            back_src_idx -= 1

            # Collect remaining notes in this chord
            while back_src_idx > 0 and abs(self.src_na['onset_sec'][back_src_idx - 1] - curr_event_time) <= CHORD_TOLERANCE:
                notes.append(copy.deepcopy(self.src_na[back_src_idx - 1]))
                back_src_idx -= 1

        return back_src_idx, self._construct_note_na(notes)


    #Since the notes themselves are passed on by the caller, the only purpose of go_back
    #is to ensure that: 1 the labels are correct, and that the timefrom/time to makes sense, since we 
    #want to keep the perf. array and the labels hidden from the midlevel.
    #For timeto, if a rollback happens, it is treated as if it were a long silence inserted
    #meaning that, timefrom doesn't change, but timeto changes. 
    #but the src pointer goes backwards. 
    #the notes would not be set with their times from performance, they would be
    #set relative to 0.
    #there is also a limitation with the labels, since it's all put as a rollback whereas
    #each note, in comparision to the score portion rolled back, should have its own label
    #but for now we move on..
    def go_back(self, src_time_to, src_time_from, notes=None, midlvl_label="rollback", repeat_index=0):
        #The variable names are hell.. time from and time to.. and src_time_from and src_time_to..
        #note that, src_time_from will be larger than src_time_to, since we are going backwards
        #same for an offset (Shift) vs a note offset.. all confusing..
        #TODO: check that src_time_to < src_time_from
       
        #add a time offset with the necessary time window for the new notes after src_time_from's equivalent
        #nearest_Idx represents src_time_from's location
        #window = (notes['onset_sec'][-1] + notes['duration_sec'][-1]) - notes['onset_sec'][0]
        
        #if notes is None:
        notes = self.get_notes_between(src_time_to, src_time_from)

        # Zero onsets and place at target time
        first_onset = notes['onset_sec'][0]
        for note in notes:
            note['onset_sec'] -= first_onset

        ordered_keys_incl_src, ordered_keys_list = self._repeat_tracker_order(src_time_from)

        if repeat_index == 0 or len(ordered_keys_incl_src) == 0:
            nearestIdx_src_time_from = np.fabs(self.time_from - src_time_from).argmin()
            nearestIdx_src_time_to = np.fabs(self.time_from - src_time_to).argmin()

            tgt_time_from = self.time_to[nearestIdx_src_time_from]
            tgt_time_to = self.time_to[nearestIdx_src_time_to]
            time_in_tgtna = self.time_to[nearestIdx_src_time_from]

            time_to_range = self.time_to[nearestIdx_src_time_to:nearestIdx_src_time_from+1].copy()
            time_from_range = self.time_from[nearestIdx_src_time_to:nearestIdx_src_time_from+1].copy()

            #save the initial src indexes to recover their ground truths
            #the key should be unique, so in theory this shouldn't crash
            #self.repeat_tracker[(tgt_time_to, tgt_time_from)] = (self.time_to[nearestIdx_src_time_to:nearestIdx_src_time_from+1],
            #                      self.time_from[nearestIdx_src_time_to:nearestIdx_src_time_from+1])

        else:
           (time_to_subarray, time_from_subarray) =  self.repeat_tracker[ordered_keys_incl_src[repeat_index - 1]]
           nearestIdx_src_time_from = np.fabs(time_from_subarray - src_time_from).argmin()
           nearestIdx_src_time_to = np.fabs(time_from_subarray - src_time_to).argmin()

           tgt_time_from = time_to_subarray[nearestIdx_src_time_from] 
           tgt_time_to = time_to_subarray[nearestIdx_src_time_to]

           time_in_tgtna = time_to_subarray[nearestIdx_src_time_from]

           time_to_range = time_to_subarray[nearestIdx_src_time_to:nearestIdx_src_time_from+1].copy()
           time_from_range = time_from_subarray[nearestIdx_src_time_to:nearestIdx_src_time_from+1].copy()
           #save the initial src indexes to recover their ground truths
           #the key should be unique, so in theory this shouldn't crash
           #self.repeat_tracker[(tgt_time_to, tgt_time_from)] = (time_to_subarray[nearestIdx_src_time_to:nearestIdx_src_time_from+1],
           #                       time_from_subarray[nearestIdx_src_time_to:nearestIdx_src_time_from+1])

        if tgt_time_from == -1 or tgt_time_to == -1:
            print('go_back: src_time_to={:.3f} or src_time_from={:.3f} maps to unmapped region (time_to=-1), skipping'.format(src_time_to, src_time_from))
            return

        tgt_time_to_apply_offset = tgt_time_from
        self._apply_warping_path_offsets(tgt_time_to_apply_offset, src_time_from - src_time_to) #we use src because we are offsetting to place the new repeat..
        self.repeat_tracker[(tgt_time_to, tgt_time_from)] = (time_to_range, time_from_range)
        #self.time_to[nearestIdx_src_time_to:] += (tgt_time_from - tgt_time_to)
        #we apply the warping path offset at the tgt_time_to_apply_offset ( nearestIdx_src_time_to) skip the old execution, so we offset by the time difference between tgt_time_from and tgt_time_to. Previously we had the offset as src_time_to(the earlier point)  and src_time_from (the later point). This would hold if the notes array is exactly the source notes with no modifications or delays or anything. what happens to the labels in that case? they should be aligned with tgt_na they should accomodate for the same shift applied. and in theory this has already been done earlier in the function

        mask = self.tgt_na['onset_sec'] >= time_in_tgtna
        self.tgt_na['onset_sec'][mask] += (src_time_from - src_time_to)

        self._shift_labels(src_time_from, (src_time_from - src_time_to), repeat_index)
        self._label_note(time_in_tgtna, time_in_tgtna + (src_time_from - src_time_to), "time_shift", midlvl_label)

        #change the grid so that src_time_to (where we want to return to) now points to our new
        #starting point (which is src_time_from)
        #append the notes at the correct time and sort
        for note in notes:
            note['onset_sec'] += time_in_tgtna

        self.tgt_na = np.concatenate((self.tgt_na, notes))

        self.tgt_na.sort(order='onset_sec') 
        return
        
    def go_fwd(self):
        #skip. 
        #the timefrom array would be set to 0 at these parts of the score that are skipped.

        return
      
    def time_offset(self, src_time, offset_time, midlvl_label, repeat_index=0): #adding silence without note insertions..
        #perhaps we shouldn't add a 1 to the index because the caller should handle making the time
        #input as the time where the offset literally would start..
        #this is just a shift in timeto, the times in the na of the perf, and the labels na.

        ordered_keys_incl_src, ordered_keys_list = self._repeat_tracker_order(src_time)

        if repeat_index == 0 or len(ordered_keys_incl_src) == 0:
            nearestIdx = np.fabs(self.time_from - src_time).argmin()
            tgt_insertion_time = self.time_to[nearestIdx]
        else:
           (time_to_subarray, time_from_subarray) =  self.repeat_tracker[ordered_keys_incl_src[repeat_index - 1]]
           nearestIdx = np.fabs(time_from_subarray - src_time).argmin()
           tgt_insertion_time = time_to_subarray[nearestIdx]

        if tgt_insertion_time == -1:
            print('time_offset: src_time {:.3f} maps to unmapped region (time_to=-1), skipping'.format(src_time))
            return

        time_in_tgtna = tgt_insertion_time
        
        mask = self.tgt_na['onset_sec'] >= time_in_tgtna
        self.tgt_na['onset_sec'][mask] += offset_time
        
        tgt_time_to_apply_offset = time_in_tgtna
        self._apply_warping_path_offsets(tgt_time_to_apply_offset, offset_time)
 
        self._shift_labels(src_time, offset_time, repeat_index)
        self._label_note(time_in_tgtna, time_in_tgtna+offset_time, "time_shift", midlvl_label)
        return
    
    def inspect_tgt(self):
        #maybe some functionality to inspect some label specific things would be helpful.
        return
    
    def get_label_miditrack(self, loc):
        midiobj = self._na_to_miditrack(self.label_na)
        midiobj.write(loc)
        return
    
    def _filter_by_label(self, name, tier='mid'): #tier could be mid or low
        if tier == 'mid':
            return self.label_na[self.label_na['midlvl_label'] == name]
        if tier == 'low':
            return self.label_na[self.label_na['lowlvl_label'] == name]
        return []
    
    #folder must already be created.
    def get_midlevel_label_miditracks(self, folder):
        #make sure the output folder exists, or create if it doesn't
        os.makedirs(folder, exist_ok=True)
        for midlvl_label in np.unique(self.label_na['midlvl_label']):
            #filter self.label_na based on the midlevel label
            #and get the low level operations that correspond to this midlevel label
            #create an na from both.
            self._na_to_miditrack(self._filter_by_label(name=midlvl_label, tier='mid')).write(os.path.join(folder, '{}.mid'.format(midlvl_label)))
        return 
    
    def _na_to_miditrack(self, na):
        midiobj = pretty_midi.PrettyMIDI()
        piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
        inst = pretty_midi.Instrument(program=piano_program)
 
        for na_note in na:
            note = pretty_midi.Note(velocity=na_note['velocity'], 
                                    pitch=na_note['pitch'], 
                                    start=na_note['onset_sec'], 
                                    end=na_note['onset_sec'] + na_note['duration_sec'])
            inst.notes.append(note)

        midiobj.instruments.append(inst)
        return midiobj
    
    def get_target_miditrack(self, loc):
        midiobj = self._na_to_miditrack(self.tgt_na)
        midiobj.write(loc)
        return
    
    def get_src_miditrack(self, loc):
        midiobj = self._na_to_miditrack(self.src_na)
        midiobj.write(loc)
        return
    
    def get_timemap(self):
        return zip(self.time_from, self.time_to)
    
    def get_repeats(self):
        #Recall that the format is: (interval for tgt time region) -> ([target time points], [src time points]), unlike time map which we make src_time -> target_time
        return self.repeat_tracker
    
    def get_adjusted_gt(self):
        #TODO: This needs to account for the repetitions.
        #check the format of the gt (needs to be a 1d array with some values)
        #we want gt to map to the tgt_na timings.

        #if the resolution of regular interpolation is insufficient, we could break it up by searching, for each annotation t, for the timefrom before and after it and interpolate based on that fragment only. 
        interpol_t = np.interp(self.ts_annot, self.time_from, self.time_to)
        
        #then, go over the repeats to handle the discontinuities:
        # for every tgt_time from to tgt_time to in the repeats
        interpol_repeats = []
        for key, val in self.repeat_tracker.items():
            tgt_time_start, tgt_time_end = key # guess those were unused because we just got the whole array
            tgt_time, src_time = val
            # get the ground truth labels corresponding to the the range src_time[0], src_time[1]
            
            #we could be pedantic and only get the annot_indexfrom to be after the src_time[0]
            # but I don't think having a repeated annotation would be a disaster.
            annot_indexfrom = np.fabs(self.ts_annot - src_time[0]).argmin()
            annot_indexto = np.fabs(self.ts_annot - src_time[-1]).argmin()

            annotations = np.interp(self.ts_annot[annot_indexfrom:annot_indexto+1], 
                                    src_time, tgt_time)
            interpol_repeats.extend(annotations)
            #interpolate every gt in source time to an equiv in that old tgt time
            #append it to the same array
        
        #sort all the labels (because we added the repeats at the end.)
        interpol_t = np.concatenate((interpol_t, np.array(interpol_repeats)))
        interpol_t.sort()

        #et. voila!
        return interpol_t
    
        #prob. not needed: if this index is within the resolution from the original time (which probably we should save in the class params)

    
