import librosa
import pretty_midi
import os
import pathlib
import csv
import ast
import numpy as np

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
    parsed_arrays = []
    metadata = []  # This will store the time and label values
    dtype = None

    with open(file_path, 'r') as file:
        for line in file:
            # Extract the time and label
            time_str, label_str, rest = line.split(',', 2)
            time_value = float(time_str.strip("[]"))
            label_value = label_str.strip()
            # Extract the params part (dictionary) from the line
            start_idx = rest.index("{")
            params_str = rest[start_idx:]
            # Convert the string representation to an actual Python dictionary
            params_dict = ast.literal_eval(params_str)
            # Extract the 'note' array and the dtype information
            note_array = params_dict['note']

            if dtype is None:
                dtype_info = str(note_array.dtype)
                dtype = parse_mistake_labels_dtype(dtype_info)

            # Convert the array to a numpy array with the correct dtype
            note_np_array = np.array(note_array, dtype=dtype)
            # Append the numpy array to the list
            parsed_arrays.append(note_np_array)
            # Append the time and label to the metadata list
            metadata.append((time_value, label_value))

    return parsed_arrays, metadata

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

class GT:
    def __init__(self, src_ts_label, tgt_ts_label):
        self.src_ts_label = np.loadtxt(src_ts_label)
        self.tgt_ts_label = np.loadtxt(tgt_ts_label)
        return
    
    def get_equiv_src_labels(self, tgt_labels, timemap, repeats):
        #check timemap for tgt
        #if failed
        #check repeats for tgt
        #repeat the matching src
        return
    
class SynmistPerformance:
    #loads the info of a single synmist performance with modified GT
    def __init__(self, src_perf, tgt_perf, mistakelabel_csv, mistake_timemap, mistakelabel_midi):
        self.tgt_mistakelabels_csv = parse_mistake_labels_file(mistakelabel_csv)
        self.tgt_mistakelabel_midi = pretty_midi.PrettyMIDI(mistakelabel_midi)
        self.tgt_performance = pretty_midi.PrettyMIDI(tgt_perf)
        self.src_perf = pretty_midi.PrettyMIDI(src_perf)
        self.mistake_timemap = csv_to_timemap(mistake_timemap)
        return
    
    def get_mistake_windows(self, recovery_buffer, mistake_types):
        #return the mistake windows
        #return around them the time of the recovery buffers
        #get the same ones in the src array too
        #get them as just time arrays
        return
    
    def get_src_equivalent(self, tgt_time):
        return
    
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
    
