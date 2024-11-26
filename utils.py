import librosa
import pretty_midi
import os
import pathlib
import csv
import ast
import numpy as np
from math import ceil, floor

#todo: function to synthesize the midi using fluidsynth and just return it as a data obj, rather than having to do these file reads/writes

def slice_prettymidi(pretty_midi_obj, start_time, end_time):
    # Create a new PrettyMIDI object
    new_pretty_midi = pretty_midi.PrettyMIDI()
    
    # Iterate over all instruments in the original PrettyMIDI object
    for instrument in pretty_midi_obj.instruments:
        # Create a new Instrument for the new PrettyMIDI object
        new_instrument = pretty_midi.Instrument(program=instrument.program, is_drum=instrument.is_drum, name=instrument.name)
        # Filter notes
        new_instrument.notes = [note for note in instrument.notes if start_time <= note.start < end_time]
        # Filter control changes
        new_instrument.control_changes = [
            cc for cc in instrument.control_changes if start_time <= cc.time < end_time]
        # Filter pitch bends
        new_instrument.pitch_bends = [
            pitch_bend for pitch_bend in instrument.pitch_bends if start_time <= pitch_bend.time < end_time]
        # Add the filtered instrument to the new PrettyMIDI object
        new_pretty_midi.instruments.append(new_instrument)
    
    # Copy over tempo changes, time signatures, and key signatures that fall within the time range
    new_pretty_midi.time_signature_changes = [
        ts for ts in pretty_midi_obj.time_signature_changes if start_time <= ts.time < end_time]
    new_pretty_midi.key_signature_changes = [
        ks for ks in pretty_midi_obj.key_signature_changes if start_time <= ks.time < end_time]
    new_pretty_midi._tick_scales = [
        ts for ts in pretty_midi_obj._tick_scales if start_time <= ts[0] < end_time]
    
    # Copy over tempo changes (you might want to filter this if there are many)
    new_pretty_midi._tempo_changes = [
        (time, tempo) for time, tempo in zip(pretty_midi_obj.get_tempo_changes()[0], pretty_midi_obj.get_tempo_changes()[1])
        if start_time <= time < end_time]
    return new_pretty_midi

#returns a dtype object that follows the dtype_str given. This is to help us serialize the saved mistakes.
#might not be needed with our new way for parsing the lines
def parse_mistake_labels_dtype(dtype_str):
    """Parse the dtype from the string."""
    dtype_items = []
    dtype_str = dtype_str[dtype_str.index("[")+1 : dtype_str.rindex("]")]
    dtype_str = dtype_str.replace("(", "").replace(")", "").split(", ")
    
    for i in range(0, len(dtype_str), 2):
        field_name = dtype_str[i].strip("'")
        field_type = dtype_str[i+1]
        dtype_items.append((field_name, field_type))
    
    return np.dtype(dtype_items)

def parse_mistake_labels_file(file_path):
    metadata = []  # This will store the time and label values
    dtype = None

    def parse_entry(entry_lines):
        # Join the accumulated lines into one single string
        entry_str = ' '.join(entry_lines)
    
        # Split the string to extract the time, label, and params
        time_part, label_part, params_part = entry_str.split(',', 2)
    
        # Clean up the time and label
        time = float(time_part.strip('[]'))
        label = label_part.strip()
    
        # Safely evaluate the params dictionary
        params_str = ast.literal_eval(params_part.strip())
        params_str = params_str.strip('"')
        #params = ast.literal_eval(params_str)
        #UNSAFE FUNCTION: 
        params = eval(params_str, {'array': np.array})
    
        # Extract 'note' and 'dtype' from params
        note = params.get('note')
        dtype = note.dtype if note is not None else None

        # Return the structured data
        return {
            'time': time,
            'label': label,
            'note': note,
            'dtype': dtype
        }

    def parse_file(file_path):
        #these are 3 parallel arrays
        dtype = None
        tgt_times = []
        tgt_labels = []
        
        with open(file_path, 'r') as file:
            header_line = next(file)
            current_entry = []
            for line in file:
                # Start accumulating a new entry when we hit a new "time, label, params"
                if line.startswith('['):
                    # Process the last entry before moving to a new one
                    if current_entry:
                        entry = parse_entry(current_entry)
                        if dtype is None:
                            dtype = entry['dtype']
                            tgt_notes= np.zeros(0, dtype=dtype)

                        tgt_notes = np.concatenate((entry['note'], tgt_notes))
                        tgt_times.append(entry['time'])
                        tgt_labels.append(entry['label'])

                    # Reset the entry accumulator
                    current_entry = [line.strip()]
                else:
                    # Continue accumulating lines for the current entry
                    current_entry.append(line.strip())
        
            # Process the last accumulated entry after the loop ends
            if current_entry:
                entry = parse_entry(current_entry)
                if dtype is None:
                    dtype = entry['dtype']
                    tgt_notes= np.zeros(0, dtype=dtype)
                            
                tgt_notes = np.concatenate((entry['note'], tgt_notes))   
                tgt_times.append(entry['time'])
                tgt_labels.append(entry['label'])
    
        tgt_notes.sort(order='onset_sec')
        return tgt_notes, np.array(tgt_times), np.array(tgt_labels)
    
    entries = parse_file(file_path)
    return entries
    

def payload_to_csv(payload, fileout):
    fields = ['time', 'label', 'params']
    with open(fileout, 'w') as csv_out:
        writer = csv.writer(csv_out)
        writer.writerow(fields)
        for row in payload:
            writer.writerow(row)
    return

def timemap_to_csv(time_map, repeats, fileout):
    fields = ['timefrom', 'timeto']
    with open(fileout, 'w') as csv_out:
        writer = csv.writer(csv_out)
        writer.writerow(fields)
        for row in time_map:
            writer.writerow(row)

        repeat_number = 1
        for (src_time_from, src_time_to), (from_times, to_times) in repeats.items():
            writer.writerow([])  # Add an empty row for clarity between repeats
            writer.writerow([f"repeat {repeat_number}:"])  # Add a header for each repeat
            
            for t_from, t_to in zip(from_times, to_times):
                writer.writerow([t_from, t_to])
            repeat_number += 1

def csv_to_timemap(filein):
    time_map = []
    repeat_tracker = {}
    current_repeat = None
    current_from_times = []
    current_to_times = []

    with open(filein, 'r') as csv_in:
        reader = csv.reader(csv_in)
        
        for row in reader:
            # Skip empty rows
            if not row:
                continue
            if row[0] == 'timefrom':
                continue
            
            # Detect repeat section
            if row[0].startswith("Repeat"):
                # If we're already in a repeat section, save the previous one
                if current_repeat:
                    repeat_tracker[current_repeat] = (current_from_times, current_to_times)
                # Start a new repeat section
                repeat_times = row[0].replace("Repeat (", "").replace(")", "").split(", ")
                current_repeat = (float(repeat_times[0]), float(repeat_times[1]))
                current_from_times = []
                current_to_times = []
            elif current_repeat:
                # Add to the current repeat section
                current_from_times.append(float(row[0]))
                current_to_times.append(float(row[1]))
            else:
                # Add to the main time_map
                time_map.append((float(row[0]), float(row[1])))
        
        # After finishing the loop, make sure to save the last repeat section
        if current_repeat:
            repeat_tracker[current_repeat] = (current_from_times, current_to_times)
    
    return time_map, repeat_tracker

#class GT: #keeps track of a time series ground truth before and after applying mistakes
#    def __init__(self, src_ts_label, tgt_ts_label):
#        self.src_ts_label = np.loadtxt(src_ts_label)
#        self.tgt_ts_label = np.loadtxt(tgt_ts_label)
#        return
    
 #   def get_equiv_src_labels(self, tgt_labels, timemap, repeats):
        #labels is an array of timestamp - 
        #check timemap for tgt
        #if failed
        #check repeats for tgt
        #repeat the matching src
#        return

def load_filenames(filename, root):
    return {
        'src_perf': os.path.join(root, '{}-src.mid'.format(filename)),
        'tgt_perf': os.path.join(root, '{}-tgt.mid'.format(filename)), 
        'mistake_timemap': os.path.join(root, '{}-mistake_timemap.csv'.format(filename)), 
        'mistakelabel_csv': os.path.join(root, '{}-label.csv'.format(filename)), 
        'mistakelabel_midi': os.path.join(root, '{}-mistake-label.mid'.format(filename)),
        'src_gt_label': os.path.join(root, '{}src-gt-label.csv'.format(filename)),
        'tgt_gt_label': os.path.join(root, '{}tgt-gt-label.csv'.format(filename))
    }
    
class SynmistPerformance:
    #loads the info of a single synmist performance with modified GT
    def __init__(self, src_perf, tgt_perf, mistakelabel_csv, mistake_timemap, mistakelabel_midi, src_gt_label, tgt_gt_label):
        (self.tgt_mistakelabel_notes, 
        self.tgt_mistakelabel_time, 
        self.tgt_mistakelabel_label) = parse_mistake_labels_file(mistakelabel_csv)
        self.tgt_mistakelabel_midi = pretty_midi.PrettyMIDI(mistakelabel_midi)
        self.tgt_performance = pretty_midi.PrettyMIDI(tgt_perf)
        self.src_perf = pretty_midi.PrettyMIDI(src_perf)
        self.mistake_timemap = csv_to_timemap(mistake_timemap)
        #to add later the src_gt and tgt_gt labels
        return
    
    def get_mistake_windows(self, recovery_buffer, mistake_types):
        mistake_centers = {}
        mistake_windows = {}
        for mistake_type in mistake_types:
            indexes = np.where(self.tgt_mistakelabel_label == mistake_type)
            mistake_centers[mistake_type] = self.tgt_mistakelabel_time[indexes]
            mistake_windows[mistake_type] = [(max(0, i-floor(recovery_buffer/2.0)), min(i+ceil(recovery_buffer/2.0), self.tgt_performance.get_end_time()))
                                             for i in mistake_centers[mistake_type]]
            
        #return the mistake windows in a list, with the mistake centered within a recovery buffer
        #return around them the time of the recovery buffers
        #get the same ones in the src array too
        #get them as just time arrays
        return
    
    #tgt_times should be an array of the time window we want to obtain a src equivalent for.
    #we should check entry by entry in the array, and get the nearest point before it if it's > 0+the threshold.
    #and return till the first entry which is within timeto+search_resolution_ms. 
    def get_src_equivalent(self, tgt_times, search_resolution_ms):
        #i think for now i'll just ignore the search_resolution_ms..
        timeto = [timeto for (timefrom, timeto) in self.mistake_timemap[0]]
        timefrom = [timefrom for (timefrom, timeto) in self.mistake_timemap[0]]

        tgt_time_out = [] 
        src_time_out = []

        for t in tgt_times:
            #check the main map and each of the repeats for the nearest index
            #each tgt time can lead to just one score time, but not vice versa. 
            closest_value = -1 #since we cannot have negative time
            closest_value_idx = None
            repeat = 0 #this is to keep track of which repeat id (or none) had the closest value.

            nearest_tgt_idx = np.fabs(t - timeto).argmin() #should be the one greater but for now whatev.
            closest_value = timeto[nearest_tgt_idx]
            closest_value_src = timefrom[nearest_tgt_idx]

            repeats = {} #holds the nearest_tgt_idx per each repeatmap

            for repeatid, repeat_map in self.mistake_timemap[1].items():
                timeto_repeat = [timeto for (timefrom, timeto) in repeat_map]
                timefrom_repeat = [timefrom for (timefrom, timeto) in repeat_map]
                repeats[repeatid] = np.fabs(t - timeto).argmin()
                if abs(t - timeto[repeats[repeatid]]) < abs(t-closest_value): 
                    closest_value = timeto_repeat[repeats[repeatid]]
                    closest_value_src = timefrom_repeat[repeats[repeatid]]
                    repeat = repeatid

            tgt_time_out.append(closest_value)
            src_time_out.append(closest_value_src)

            #at the end of this,
            #repeat would tell whether this is from a repeated part or not. 0 if not, yes if greater
            #and repeats[repeatid] tells which is the closest index to t we are trying to find.
            #this code can definitely be more efficient, but i'm sticking to this mediocrity for now.

        return tgt_time_out, src_time_out
    
    def get_src_data(self, src_times):
        
        return
    
    def get_tgt_time(self, tgt_times):
        #to be used twice, once to get the tgt times, and then the recovery times, all from get mistake windows
        return

def synthesize_midi(score_path, fs=44100):
    score_synth = pretty_midi.PrettyMIDI(score_path).fluidsynth(fs=fs)
    return score_synth


def save_global_stats(run_path):
    #the dataframe is saved in same runpath
    #this payload saving function needs to change a bit, but for now we will just read from it as is.

    return

def get_window_info(stem, seconds_start, seconds_end):
    #this function returns the info of this window
    #get window info can be implemented with get_mistake_info
    #check the midi label track
    #check the src track
    #check the target track
    #print the src with red lines in the mistake insertion points
    #print the target with note additions in red? (or just the span of the mistake)
    #return a list of mistakes and their start points
    #should return the length of contiguous mistake areas for the data.
    return

def get_mistake_info(stem, seconds_start, note=None):
    #when time and note are given, return the saved parameters relating to that error

    #
    return
    
