import partitura as pt
import mido
import math
import numpy as np
import copy
import pretty_midi

#MIDI Locations for the labels so that we can decipher what's being output.

MID_FWDBCKWD = 21      #A0
MID_ROLLBACK = 25      #C#1
MID_MISTOUCH = 28      #E1
MID_WRNG_PRED_INS = 30 #F#1
MID_DRAG = 33          #A1

LOW_INSERT = 1 #C#0
LOW_DELETE = 3 #D#0
LOW_GOBACK = 5 #F0
LOW_SHIFT = 7  #G0
LOW_GOBACK = 9 #A#0

DEFAULT_MID = 20 #G#2
DEFAULT_LOW = 0  #C-1

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
    'go_back': LOW_GOBACK
}

LABEL_VELOCITY = 10

#We cannot have enough control as in the case of m-san's code to the temporal shifts first. 
#In Akira's logic, the 'scheduling' happens with the timeshifts at the same time, making things
        #significantly easier.

label_na_fields = [('onset_sec', '<f4'), ('duration_sec', '<f4'), ('pitch', '<i4'), ('velocity', '<i4'), 
              ('midlvl_label', '<U256'), ('lowlvl_label', '<U256')]
regular_na_fields = [('onset_sec', '<f4'), ('duration_sec', '<f4'), ('onset_tick', '<i4'), ('duration_tick', '<i4'), 
                     ('pitch', '<i4'), ('velocity', '<i4'), ('track', '<i4'), ('channel', '<i4'), 
                     ('id', '<U256')]

#order of applying the mistakes:
#time based
#then, if several at the same time: insert, delete, goback, offset,
#actually it might not matter so much.. try without sorting first. 

#btw if we want to delete after an insertion, and we give the same 'time' as the insertion, we should
#make the extra step to check that we won't delete what we just inserted. (or not?). Anyway, ids would work 
#in this case.

#food for thought, should the label be the same for low level and mid level stuff?

class lowlvl:
    def __init__(self, src_na):
        #the labels would be maintained wrt to timeto or the tgt_na 
        #target = using the nearest giventime, get the timeto. In thoery, all existing score notes will
        #have a timefrom value in the safe zones.
        #TODO: if src is an empty note array, fail and exit

        start = src_na['onset_sec'][0]
        end = src_na['onset_sec'][-1]+ src_na['onset_sec'][-1]
        self.time_from = np.linspace(start, end, int(math.ceil(end)-math.ceil(start))*20) #not sure of resolution.. 
        self.time_to = self.time_from.copy()
        
        self.onsets = src_na['onset_sec']
        self.src_na = src_na
        self.tgt_na = copy.deepcopy(self.src_na)
        self.label_na = np.zeros(0, dtype=label_na_fields)

        self.time_res = 0.05 #mostly used for the time_offset calculation.. let's see.

        #This index points to the note item you are standing at
        self.src_idx = 0
        self.tgt_idx = 0
        return
    
    def _label_note(self, start, end, lowlvl_label, midlvl_label):
        mid_label_pitch = mid_label_pitch_map[midlvl_label] if midlvl_label in mid_label_pitch_map else DEFAULT_MID
        low_label_pitch = low_label_pitch_map[lowlvl_label] if lowlvl_label in low_label_pitch_map else DEFAULT_LOW

        #add 2 notes, one mid and one low.
        new_label = np.array([(start, end-start, mid_label_pitch, LABEL_VELOCITY, lowlvl_label, midlvl_label), 
                              (start, end-start, low_label_pitch, LABEL_VELOCITY, lowlvl_label, midlvl_label)], dtype=self.label_na.dtype)
        self.label_na = np.concatenate((new_label, self.label_na))
        self.label_na.sort(order='onset_sec')
        return 
    
    def _shift_labels(self, src_time, offset):
        #shift all labels after time 'time' by offset s
        #only used in case we relax the requirement that changes have to be done from earliest to latest.
        #offset can be +ve or -ve.
        nearestIdx = np.fabs(self.time_from - src_time).argmin()
        time_in_tgtna = self.time_to[nearestIdx]

        labels_start = self.label_na['onset_sec']
        label_na_idx = np.fabs([labels_start - time_in_tgtna]).argmin() #should be the one greater but for now whatev.
        self.label_na[label_na_idx:]['onset_sec'] += offset
        return
    
    #this in itself does not shift consequent pitches
    def pitch_insert(self, src_time, pitch, duration, velocity, midlvl_label):
        #TODO: find a calculate ticks option from partitura.. 
        #TODO: from the match file, find what id is placed for notes that are 'extra'

        #get corresponding insertion time in tgt_na.
        nearestIdx = np.fabs(self.time_from - src_time).argmin()
        tgt_insertion_time = self.time_to[nearestIdx]

        #following the regular_na fields structure above..
        new_note = np.array([(tgt_insertion_time, duration, 0, 0, pitch, velocity, 0, 0, 'none')], dtype=self.tgt_na.dtype)
        self.tgt_na = np.concatenate((self.tgt_na, new_note))

        self.tgt_na.sort(order='onset_sec')
        self._label_note(tgt_insertion_time, tgt_insertion_time +duration, midlvl_label, 'pitch_insert')
        return
    
    def _find_note_in_tgt(self, src_time, pitch):
        nearestIdx = np.fabs(self.time_from - src_time).argmin()
        time_in_tgtna = self.time_to[nearestIdx]

        window = 0.05 #a window 50 ms before and after for trying to find the onset of the pitch in question.

        #we could evade little time offset problems by:
        #1. setting a tol. window around the src_time (which we do) which we check for all notes of the specified
        # pitch and get the nearest one to the given time.
        #2. find all notes of that pitch in the score and just choose the nearest one within a thresh.
        lowerbound_in_tgt_na = np.fabs(np.array([i['onset_sec'] for i in self.tgt_na]) - (time_in_tgtna - window)).argmin()
        upperbound_in_tgt_na = np.fabs(np.array([i['onset_sec'] for i in self.tgt_na]) - (time_in_tgtna + window)).argmin()

        found = False
        note_options = []

        for i in range(lowerbound_in_tgt_na, upperbound_in_tgt_na+1):
            #find all notes with the corresponding pitch
            if self.tgt_na['pitch'][i] == pitch:
                note_options.append((self.tgt_na['onset_sec'][i], i))
                found = True
                
        if found==False:                                                                                    
            print('pitch not found')  
            return False, None, None
        else:
            #find closest to the src_time from the list. should be at the top
            note_start, note_idx = sorted(note_options, key=lambda x: np.fabs(x[0] - src_time))[0]

        return found, note_idx, note_start

    
    def pitch_delete(self, src_time, pitch, midlvl_label):
        found, note_idx, note_start = self._find_note_in_tgt(src_time, pitch)

        if not found:
            return
        else:
            note_end = note_start + self.tgt_na['duration_sec'][note_idx]
            self.tgt_na = self.tgt_na[~((self.tgt_na['onset_sec'] == self.tgt_na['onset_sec'][note_idx]) & 
                                        (self.tgt_na['pitch'] == self.tgt_na['pitch'][note_idx]))]

            self._label_note(note_start, note_end, 'pitch_delete', midlvl_label)          
        return
    
    #works for shorten note and extend note
    def change_note_offset(self, src_time, pitch, offset_shift, midlvl_label):
        found, note_idx, note_start = self._find_note_in_tgt(src_time, pitch)
        #it should always be found.. unless the note has been deleted in prior processing. 
        if not found:
            return
        note_end = note_start + self.tgt_na['duration_sec'][note_idx]
        self.tgt_na[note_idx]['duration_sec'] += offset_shift
        self._label_note(note_start, note_end, "change_offset", midlvl_label)
        return
    
    def change_note_onset(self, src_time, pitch, onset_shift, midlvl_label, lowlvl_label):
        #this would have a limit, because if it goes tooooo far, then it should be a deleted note in
        #this region, and a note insertion in the other..
        #this is for wiggles that make sense, without the need to change time_from or time_to.
        return
    
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
    def go_back(self, src_time, back_time, notes=[]):
        #notes should be with the same dtype and fields as na
        #add an offset after src_time, which is presumably the note at which the playhead is
        #rollback. if i say i want to roll back from a point, and this note coincides with a note onset, probably we need to play these notes before doing the rollback.
        #meaning that we might have to take into account that the notes are inserted after the notes end, 
        #currently the logic would probably just add them starting the src_time. 
        shift_duration = notes['onset_sec'][-1] + notes['duration_sec'][-1] #corresponding to onset + duration
        
        self.offset(src_time, shift_duration, "rollback")

        #starting performance na time:
        nearest_time_Idx = np.fabs(self.time_from - src_time).argmin()
        time_in_tgtna = self.time_to[nearest_time_Idx]

        #in theory, since the offset is done, all we should do is append the notes at the correct time
        #and sort
        for note in notes:
            note['onset_sec'] += time_in_tgtna
            #make sure that notes fits the dtype of self.tgt_na.dtype
        
        self.tgt_na = np.concatenate((self.tgt_na, notes))
        self.tgt_na.sort(order='onset_sec') 
        return
        
    def go_fwd(self):
        #skip. 
        #the timefrom array would be set to 0 at these parts of the score that are skipped.

        return
    
    
    def time_offset(self, src_time, offset_time, midlvl_label): #adding silence without note insertions..
        #perhaps we shouldn't add a 1 to the index because the caller should handle making the time
        #input as the time where the offset literally would start..
        #this is just a shift in timeto, the times in the na of the perf, and the labels na.
        nearestIdx = np.fabs(self.time_from - src_time).argmin()
        time_in_tgtna = self.time_to[nearestIdx]
        
        onsets = self.tgt_na['onset_sec']
        tgt_na_idx = np.fabs([onsets - time_in_tgtna]).argmin() #not sure if it should be one greater.. look out
        self.tgt_na[tgt_na_idx:]['onset_sec'] += offset_time

        self.time_to[nearestIdx:] += offset_time       
        self._shift_labels(src_time, offset_time)
        self._label_note(time_in_tgtna, offset_time, "time_offset", midlvl_label)
        return
    
    def inspect_tgt(self):
        #maybe some functionality to inspect some label specific things would be helpful.
        return
    
    def get_label_miditrack(self, loc):
        midiobj = self._na_to_miditrack(self.label_na)
        midiobj.write(loc)
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
    
