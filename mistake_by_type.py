
import partitura as pt
import numpy as np
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

        self.performance = pt.load_performance(performance_path)

        rc = RegionClassifier(performance_path, save=False)

        self.na = rc.na     # note array with region classification

        self.forward_backward_insertion(self.na[self.na['id'] == 'n17'])

        # pt.save_performance_midi(self.performance, performance_path[:-4] + "_mk.mid")
        # hook()


    def paint_velocity(self, velocity_map):
        """paint the velocity of the notes in piece, according to the dict that maps 
            column attribute to a specific velocity.  

            Note that you want to prioritize the keys in the velocity map (from low to high), 
                as the former ones will be overwritten by later ones.

            velocity_map: {"is_double_note": 64, "is_scale_note": 127,} 
        """

        for note in self.performance.performedparts[0].notes:
            note['velocity'] = 1

            note_in_na = self.na[self.na['id'] == note['id']]

            for key in velocity_map.keys():
                if note_in_na[key]:
                    note['velocity'] = velocity_map[key]

        return 

    ########### Functions for adding various mistakes ############

    def forward_backward_insertion(self, note, forward=True):
        """insert notes that belongs to the previous / later onset

    
        """

        # forward: search for the nearest previous neighbor and double it
        if forward:
            insert_pitch = self.na[self.na['offset_sec'] <= note['onset_sec']][-1]

        notes_cpy = copy.deepcopy(self.performance.performedparts[0].notes)
        for nn in self.performance.performedparts[0].notes:
            if nn['id'] == note['id']:
                nn_cpy = copy.deepcopy(nn)
                nn_cpy['pitch'] = insert_pitch['pitch']
                nn_cpy['note_on'] = nn['note_on'] + np.random.random() * 0.05
                nn_cpy['velocity'] = int(np.random.random() * nn['velocity'])
                notes_cpy.append(nn_cpy)
                hook()
            


        return 



if __name__ == '__main__':

    mk = Mistaker("/Users/huanzhang/01Acdemics/PhD/Research/Datasets/Burgmuller/b-04-annot.mid", burgmuller=True)