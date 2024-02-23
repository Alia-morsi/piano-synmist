import partitura as pt
import mido
import math
import numpy as np
import copy

NA_ONSET_idx = 0
NA_DUR_idx = 1

MIDLVL_INSERT = 1
MIDLVL_ROLLBACK = 2
MIDLVL_WRNG_PRED_INS = 3
LABEL_PITCH = 0
LABEL_VELOCITY = 20


#We cannot have enough control as in the case of m-san's code to the temporal shifts first. 
#In Akira's logic, the 'scheduling' happens with the timeshifts at the same time, making things
        #significantly easier.

label_na_fields = [('onset_sec', '<f4'), ('duration_sec', '<f4'), ('pitch', '<i4'), ('velocity', '<i4'), 
              ('midlvl_label', '<U256'), ('lowlvl_label', '<U256')]
regular_na_fields = [('onset_sec', '<f4'), ('duration_sec', '<f4'), ('onset_tick', '<i4'), ('duration_tick', '<i4'), 
                     ('pitch', '<i4'), ('velocity', '<i4'), ('track', '<i4'), ('channel', '<i4'), 
                     ('id', '<U256')]

#food for thought, should the label be the same for low level and mid level stuff?

class lowlvl:
    def __init__(self, src_na):
        #the labels would be maintained wrt to timeto or the tgt_na 
        #target = using the nearest giventime, get the timeto. In thoery, all existing score notes will
        #have a timefrom value in the safe zones.

        #TODO: if src is an empty note array, fail and exit
        start = src_na[0][NA_ONSET_idx]
        end = start + src_na[-1][NA_DUR_idx]
        self.timefrom = np.linspace(start, end, int(math.ceil(end)-math.ceil(start))*20) #not sure of resolution.. 
        self.timeto = self.timefrom.copy()
        
        self.onsets = [note[NA_ONSET_idx] for note in src_na]
        self.src_na = src_na
        self.tgt_na = copy.deepcopy(self.src_na)
        self.label_na = np.zeros(0, dtype=label_na_fields)

        #This index points to the note item you are standing at
        self.src_idx = 0
        self.tgt_idx = 0
        return
    
    def _label_note(self, start, end, lowlvl_label, midlvl_label):
        #since there is no notearray to midiroll functionality in partitura, 
        #and the notearray is literally a note array, we just remove the fields that dont make sense.
        self.label_na.append((start, end-start, LABEL_PITCH, LABEL_VELOCITY, lowlvl_label, midlvl_label))
        self.label_na.sort()
        return 
    
    def _shift_labels(self, time, offset):
        #shift all labels after time 'time' by offset s
        #only used in case we relax the requirement that changes have to be done from earliest to latest.
        #offset can be +ve or -ve.
        labels_start = np.array([i['onset_sec'] for i in self.labels_na])
        nearestIdx = np.fabs([labels_start - time]).argmin() #should be the one greater but for now whatev.
        self.labels_na[nearestIdx:]['onset_sec'] += offset
        self.labels_na[nearestIdx:]['duration_sec'] += offset
        return
    
    #this in itself does not shift consequent pitches
    def pitch_insert(self, src_time, pitch, duration, velocity, performed_parts, midlvl_label, lowlvl_label):
        #TODO: find a calculate ticks option from partitura.. 
        #TODO: from the match file, find what id is placed for notes that are 'extra'

        #get corresponding insertion time in tgt_na.
        nearestIdx = np.fabs(self.time_from - src_time).argmin()
        tgt_insertion_time = self.time_to[nearestIdx]

        #following the regular_na fields structure above..
        na_note = (tgt_insertion_time, duration, 0, 0, pitch, velocity, 0, 0)
        self.tgt_na.append(na_note)
        self.tgt_na.sort(key = lambda x: x['start'])

        self._label_note(tgt_insertion_time, tgt_insertion_time +duration, midlvl_label, lowlvl_label)

        return
    
    def pitch_delete(self, src_time, pitch, midlvl_label, lowlvl_label):
        nearestIdx = np.fabs(self.time_from - src_time).argmin()
        time_in_tgtna = self.time_to[nearestIdx]

        window = 0.05 #a window 50 ms before and after for trying to find the onset of the pitch in question.

        #another approach could be to just get all instances of this pitch, and get the nearest one to the 
        #expected time..
        lowerbound_in_tgt_na = np.fabs(np.array([i['onset_sec'] for i in self.tgt_na]) - time_in_tgtna - 0.05).argmin()
        upperbound_in_tgt_na = np.fabs(np.array([i['onset_sec'] for i in self.tgt_na]) - time_in_tgtna + 0.05).argmin()

        found = False
        note_options = np.array([])
        for i in range(lowerbound_in_tgt_na, upperbound_in_tgt_na +1):
            #find all notes with the corresponding pitch
            if self.tgt_na[i].pitch == pitch:
                note_options.append((self.tgt_na[i][0], i))
                found = True

        if found==False:
            print('pitch not found')  
        else:
            #find closest to the src_time from the list. should be at the top
            note_idx = np.sort(note_options, key=lambda x: np.fabs(x[0] - src_time))[0]
            note_start = self.tgt_na[note_idx][0]
            note_end = note_start + self.tgt_na[note_idx][1]
            #remove it
            self.tgt_na.pop(note_idx)

            self._label_note(note_start, note_end, lowlvl_label, midlvl_label)          
        return
    
    #works for shorten note and extend note
    def change_note_offset(self, src_time, pitch, offset_shift, midlvl_label, lowlvl_label):
        return
    
    def change_note_onset(self, src_time, pitch, onset_shift, midlvl_label, lowlvl_label):
        #this would have a limit, because if it goes tooooo far, then it should be a deleted note in
        #this region, and a note insertion in the other..
        #this is for wiggles that make sense, without the need to change time_from or time_to.
        return
    
    #Since the notes themselves are passed on by the caller, the only purpose of go back
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
        #add an offset after src_time, which is presumably the note at which the playhead is
        shift_duration = notes[:-1][0] + notes[:-1][1] #corresponding to onset + duration
        self.offset(src_time, shift_duration, "rollback")

        #starting performance na time:
        nearest_time_Idx = np.fabs(self.time_from - src_time).argmin()
        time_in_tgtna = self.time_to[nearest_time_Idx]

        #in theory, since the offset is done, all we should do is append the notes at the correct time
        #and sort
        for note in notes:
            note['onset_sec'] += time_in_tgtna
            self.tgt_na.append()
        return
        
    def go_fwd(self):
        #skip. 
        #the timefrom array would be set to 0 at these parts of the score that are skipped.

        return
    
    def offset(self, src_time, offset_time, midlvl_label, lowlvl_label='offset'): #adding silence without note insertions..
        #this is just a shift in timeto, the times in the na of the perf, and the labels na.
        nearestIdx_in_time= np.fabs(self.time_from - src_time).argmin()
        time_in_tgtna = self.time_to[nearestIdx_in_time]
        nearestIdx_in_tgt_na = np.fabs(np.array([i['onset_sec'] for i in self.tgt_na]) - time_in_tgtna).argmin()

        self.tgt_na[nearestIdx_in_tgt_na:]['onset_sec'] += offset_time
        self.tgt_na[nearestIdx_in_tgt_na:]['offset_sec'] += offset_time

        self.time_to[nearestIdx_in_time:] += offset_time       
        self._shift_labels(src_time, offset_time)
        self._label_note(time_in_tgtna, offset_time, midlvl_label, lowlvl_label)

        return
    
    def _get_label_miditrack(self):
        for n in self.label_na:
            #create a track (or more, depending on level) with prettymidi or mido..
            print('creating miditrack from labels')
        return
    
