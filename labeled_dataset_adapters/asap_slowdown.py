import pretty_midi
import os
import sys
import glob


tempo_slowdown_mapping = {
    (0, 60): 10,   # 0-60 BPM -> 10% slowdown
    (61, 120): 20,  # 61-120 BPM -> 20% slowdown
    (121, 180): 30, # 121-180 BPM -> 30% slowdown
    (181, 240): 40  # 181-240 BPM -> 40% slowdown
}

def get_tempo_at_time(time, tempo_times, tempi):
    for i in range(len(tempo_times) - 1):
        if tempo_times[i] <= time < tempo_times[i + 1]:
            return tempi[i]
    return tempi[-1] 

def get_slowdown_percentage(tempo, tempo_slowdown_mapping):
    for tempo_range, slowdown_percentage in tempo_slowdown_mapping.items():
        if tempo_range[0] <= tempo <= tempo_range[1]:
            return slowdown_percentage
    return 0 

def slow_down_midi(input_file, output_file, tempo_slowdown_mapping):
    midi_data = pretty_midi.PrettyMIDI(input_file)
    tempo_times, tempi = midi_data.get_tempo_changes()
    
    applied_slowdowns = []  # To store the applied slowdowns for each note/event

    for instrument in midi_data.instruments:
        for note in instrument.notes:
            tempo = get_tempo_at_time(note.start, tempo_times, tempi)
            slowdown_percentage = get_slowdown_percentage(tempo, tempo_slowdown_mapping)
            slowdown_factor = 1 + slowdown_percentage / 100.0
            note.start *= slowdown_factor
            note.end *= slowdown_factor
            applied_slowdowns.append((note.start, slowdown_percentage))
        for bend in instrument.pitch_bends:
            tempo = get_tempo_at_time(bend.time, tempo_times, tempi)
            slowdown_percentage = get_slowdown_percentage(tempo, tempo_slowdown_mapping)
            slowdown_factor = 1 + slowdown_percentage / 100.0
            bend.time *= slowdown_factor
            applied_slowdowns.append((bend.time, slowdown_percentage))
        for cc in instrument.control_changes:
            tempo = get_tempo_at_time(cc.time, tempo_times, tempi)
            slowdown_percentage = get_slowdown_percentage(tempo, tempo_slowdown_mapping)
            slowdown_factor = 1 + slowdown_percentage / 100.0
            cc.time *= slowdown_factor
            applied_slowdowns.append((cc.time, slowdown_percentage))
    
    # Adjust the timing of all tempo changes
    adjusted_tempo_times = [time * (1 + get_slowdown_percentage(get_tempo_at_time(time, tempo_times, tempi), tempo_slowdown_mapping) / 100.0) for time in tempo_times]
    
    #what about the tempo change value...
    for tempo_time, adjusted_tempo_time in zip(tempo_times, adjusted_tempo_times): 
        if tempo_time == 0:
            continue
        midi_data.adjust_times([tempo_time], [adjusted_tempo_time])    
    
    midi_data.write(output_file)
    
    return applied_slowdowns


def slow_down_annotations(input_file, output_file, applied_slowdowns):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                start_time = float(parts[0])
                end_time = float(parts[1])
                
                # Find the corresponding slowdown percentage
                # Find the closest applicable slowdown percentage
                slowdown_percentage = 0
                for (time, percentage) in applied_slowdowns:
                    if start_time >= time:
                        slowdown_percentage = percentage
                    else:
                        break

                slowdown_factor = 1 + slowdown_percentage / 100.0
                
                # Adjust the times
                start_time *= slowdown_factor
                end_time *= slowdown_factor
                
                # Write the adjusted times and original annotations
                outfile.write(f"{start_time}\t{end_time}\t" + '\t'.join(parts[2:]) + "\n")
            else:
                outfile.write(line)

def slow_down_midi_bak(input_file, output_file, slowdown_percentage):
    # Load the MIDI file
    midi_data = pretty_midi.PrettyMIDI(input_file)
    
    # Calculate the slowdown factor
    slowdown_factor = 1 + slowdown_percentage / 100.0
    
    # Adjust the timing of all notes and events
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            note.start *= slowdown_factor
            note.end *= slowdown_factor
        for bend in instrument.pitch_bends:
            bend.time *= slowdown_factor
        for cc in instrument.control_changes:
            cc.time *= slowdown_factor
        
    # Adjust the timing of all tempo changes
    
    for tempo_change in midi_data.get_tempo_changes()[0]:
        if tempo_change == 0:
            continue
        midi_data.adjust_times([tempo_change], [tempo_change * slowdown_factor])
    
    # Save the modified MIDI file
    midi_data.write(output_file)

def slow_down_annotations_bak(input_file, output_file, slowdown_percentage):
    # Calculate the slowdown factor
    slowdown_factor = 1 + slowdown_percentage / 100.0
    
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                # Adjust the first two time columns
                start_time = float(parts[0]) * slowdown_factor
                end_time = float(parts[1]) * slowdown_factor
                # Write the adjusted times and original annotations
                outfile.write(f"{start_time}\t{end_time}\t" + '\t'.join(parts[2:]) + "\n")
            else:
                # Just in case there are lines that do not match the expected format
                outfile.write(line)

def process_files(input_dir, output_dir, slowdown_percentage):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Process all MIDI files in the input directory
    #the globbing might be the cause of a bug in the future..
    for midi_file in glob.glob(os.path.join(input_dir, "*.mid")):
        filename = os.path.basename(midi_file)
        output_midi_file = os.path.join(output_dir, filename)
        applied_slowdowns = slow_down_midi(midi_file, output_midi_file, tempo_slowdown_mapping)
        print(f"Processed MIDI file: {filename}")

        annotation_file = os.path.join(input_dir, filename.replace(".mid", "_annotations.txt"))
        if os.path.exists(annotation_file):
            output_annotation_file = os.path.join(output_dir, filename.replace(".mid", "_annotations.txt"))
            slow_down_annotations(annotation_file, output_annotation_file, applied_slowdowns)
            print(f"Processed annotation file: {annotation_file}")
    

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: python slow_down_midi.py <input_midi_file_or_directory> <output_file_or_directory> <slowdown_percentage> <input_annotation_file_or_directory> <output_annotation_file_or_directory>")
        sys.exit(1)

    input_midi_path = sys.argv[1]
    output_midi_path = sys.argv[2]
    input_annotation_path = sys.argv[3]
    output_annotation_path = sys.argv[4]

    if os.path.isdir(input_midi_path) and os.path.isdir(input_annotation_path):
        process_files(input_midi_path, output_midi_path, tempo_slowdown_mapping)
    else:
        #slow_down_midi(input_midi_path, output_midi_path, slowdown_percentage)
        #slow_down_annotations(input_annotation_path, output_annotation_path, slowdown_percentage)
        applied_slowdowns = slow_down_midi(input_midi_path, output_midi_path, tempo_slowdown_mapping)
        slow_down_annotations(input_annotation_path, output_annotation_path, applied_slowdowns)

