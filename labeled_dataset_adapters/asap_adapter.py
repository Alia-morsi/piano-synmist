import torch
from torch.utils.data import Dataset
import os
import numpy as np
import glob

"""
        Fetches the ground truth (time labels) from the appropriate
        label file. Then, quantises it to the nearest spectrogram frames in
        order to allow fair performance evaluation.
"""

class ASAPLoader(Dataset):
    def __init__(self, dataset_root, downbeat=True):
        self.dataset_root = dataset_root
        #check if the path is correct
        self.downbeat = downbeat
        self.data_names, self.data_labelpaths = self._get_data_list()

    def __len__(self):
        return len(self.data_names)

    def __getitem__(self, i):
        "Overloading the square brackets to return the name, path, and labels. This is not meant"
        "for deep learning so we're not sticking to the return formats expected"
        "this should return the path of the midi file and the ts array"
        return os.path.join(self.data_labelpaths[i], '{}.mid'.format(self.data_names[i])), self._load_labels(i)[1]

    def _text_label_to_float(self, text):
        """Exracts beat time from a text line and converts to a float"""
        """ adapted to the asap dataset format """
        allowed = '1234567890. \t'
        #this can be done in a smarter way using the asap_annotations.json file, but for now let's keep it the same way
        #here we will just turn the db into 2 and the b into 1. not sure if this is a good way..
        
        filtered = ''.join([c for c in text if c in allowed])
        if '\t' in filtered:
            t = filtered.rstrip('\n').split('\t')
        else:
            t = filtered.rstrip('\n').split(' ')
        
        #remove extra info from keychange or timechange lines
        if ',' in t[2]:
            t[2] = t[2].split(',')[0]
            
        val = 1.0
        if t[2] == 'b':
            val = 2
        elif t[2] == 'db':
            val = 1
        else: #in case of bR, tho this numbering might not make sense at all..
            val = 3

        return float(t[1]), float(val) #t[0] and t[1] are time, and t[2] is db or b

    def _load_labels(self, i):
        """
        Given an index for the data name array, return the filename and label dumps.
        """
        data_name = self.data_names[i]

        with open(
                os.path.join(self.data_labelpaths[i], data_name + '_annotations.txt'),
                'r') as f:
            beat_floats = []
            beat_indices = []

            for line in f:
                parsed = self._text_label_to_float(line)
                beat_floats.append(parsed[0])
                beat_indices.append(parsed[1])
            beat_times = np.array(beat_floats)

            #if not self.downbeats: #probably this will be working funny rn because of my gt scheme of 1, 0.5, and 2..
            #   downbeat_times = np.array(
            #        [t for t, i in zip(beat_floats, beat_indices) if i == 1])
            
        #for now just keep the beat floats
        return data_name, beat_times


    def _get_data_list(self):
        """ Fetches list of datapoints.. be wary of mistmatches in numbers between modalities"""
        """ This is a very dataset specific part of the code"""
 
        #find all the leaves that have a ._annotation.text
        # their names would be formed of the whole path, which would be the same between the spec. and the label dirs. 
        # there might be more.txt files than spectrogram dirs because some of the asap files are only in midi (not audio) 

        #small modification to the code to accept folders with varying structures
        midis = glob.glob('{}/**/*.mid'.format(self.dataset_root), recursive=True)

        labelpaths = [os.path.dirname(i) for i in midis]
        names = [os.path.splitext(os.path.split(i)[1])[0] for i in midis] 

        return names, labelpaths
