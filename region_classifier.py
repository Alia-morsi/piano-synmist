import partitura as pt
import numpy as np
import numpy.lib.recfunctions as rfn
import hook

class RegionClassifier():
    def __init__(self, performance_path, burgmuller=False, save=True):
        """This is the functionality for detecting / parsing regions of interest, 
        based on the assumption that errors are usually associated with certain 
        regions / technique groups and the probability of making mistakes varies 
        depending on the musical context.

        Args:
            performance_path (str): 
        """

        self.performance = pt.load_performance(performance_path)

        # remove all the pitch = 0,1 notes for burgmuller
        if burgmuller:
            self.performance.performedparts = self.performance.performedparts[:1]
        

        # self.na will be noted according to the regions and the performance will be modified according to the na
        self.na = self.performance.note_array()
        self.na = rfn.append_fields(self.na, "offset_sec", self.na['onset_sec'] + self.na['duration_sec'], usemask=False)

        # Remove 'P00_' from the 'col1' column
        modified_col = np.char.replace(self.na['id'], 'P00_', '')
        self.na = rfn.drop_fields(self.na, 'id')
        self.na = rfn.append_fields(self.na, 'id', modified_col, usemask=False)

        self.double_note_detection()
        self.scale_note_detection()
        self.block_chords_note_detection()

        self.na = rfn.append_fields(self.na, 'others', [
            int(not (self.na[i]['is_double_note'] or self.na[i]['is_scale_note'] or self.na[i]['is_block_chords_note'])) for i in range(len(self.na))], usemask=False)

        self.paint_velocity({
            "is_double_note": 64,
            "is_scale_note": 92,
            "is_block_chords_note": 127,
        })

        if save:
            pt.save_performance_midi(self.performance, performance_path[:-4] + "_rc.mid")


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

    ########### Functions for region detection and helpers ############

    def double_note_detection(self):
        """take the note array and label 1 for notes that's possibly belong to a double note stream. 
        
        Onset-diameter-neighbor approach, for each note search for a diameter (eg. 5) to see if there is a 
            same-onset-neighbor (with in a threshold like 50ms). With exactly 2 then it's likely to be a double
            note structure. 
        After identifying the possible double note, we take a clustering based on the proximity of timing. For 
            the clustered notes they would be put into a double note stream.

        """

        neighbors_len = np.array([self.onset_neighbor_num(row) for row in self.na])
        double_notes = self.na[neighbors_len > 0]

        self.na = rfn.append_fields(self.na, 'is_double_note', [int(l == 1) for l in neighbors_len], usemask=False)

        return 
    
    def scale_note_detection(self):
        """take the note array and label 1 for notes that's possibly belong to a running scale

        TODO: check for ongoing direction?
        
        Consecutive neighbors approach: If the note has 1 previous neighbor and 1 next neighbor 
            within 2 semitones, it's likely to be inside the scale. Note that this doesn't include
            the beginning or the end of scales, but we just need a rough estimate.
        """

        neighbors_len = np.array([self.consecutive_neighbor_num(row) for row in self.na])

        self.na = rfn.append_fields(self.na, 'is_scale_note', [min(min(1, p), min(1, n)) for p, n in neighbors_len], usemask=False)

        return 

    def block_chords_note_detection(self):
        """take the note array and label 1 for notes that's possibly belong to a chordal block
        
        Parallel neighbors approach: If there are at least 2 other notes with simultaneous onset 
            and offset (within threshold), then it's likely to be a block chord
        """
        neighbors_len = np.array([self.onset_offset_neighbor_num(row, remove_chord_outlier=True) for row in self.na])

        self.na = rfn.append_fields(self.na, 'is_block_chords_note', [int(l >= 2) for l in neighbors_len], usemask=False)

        return 

    def arpeggios_detection():
        return


    def onset_neighbor_num(self, note, diameter=5, threshold=0.05):
        """helper function for getting the numbers of onset neighbor. 

        Args:
            note (np.array): a row of structured array with note information
            diameter (int, optional): The amount of semitones to look for. Defaults to 5.
            threshold (float, optional): Timing deviation threshold for finding the neighbor. Defaults to 0.05.
        """

        same_onset = self.na[np.abs(self.na['onset_sec'] - note['onset_sec']) <= threshold]
        neighbors = same_onset[np.abs(same_onset['pitch'] - note['pitch']) <= diameter]

        neighbors = neighbors[neighbors['id'] != note['id']]
        return len(neighbors)

    def onset_offset_neighbor_num(self, note, threshold=0.05, remove_chord_outlier=False):
        """helper function for getting the numbers of onset and offset neighbor (parallel neighbor). 
            no restriction on semitones diameter.

        Args:
            note (np.array): a row of structured array with note information
            threshold (float, optional): Timing deviation threshold for finding the neighbor. Defaults to 0.05.
        """

        same_onset = self.na[np.abs(self.na['onset_sec'] - note['onset_sec']) <= threshold]
        same_offset = same_onset[np.abs(same_onset['offset_sec'] - note['offset_sec']) <= threshold]

        neighbors = same_offset[same_offset['id'] != note['id']]

        # if detecting chords, exlude the outlier that's too distant from the other notes - neighbors = 0
        if len(neighbors) and remove_chord_outlier:
            if np.abs(neighbors['pitch'] - note['pitch']).min() >= 12:
                return 0
        
        return len(neighbors)


    def consecutive_neighbor_num(self, note, diameter=2, threshold=0.05):
        """helper function for getting the numbers of consecutive neighbor. 

        Args:
            note (np.array): a row of structured array with note information
            diameter (int, optional): The amount of semitones to look at. Defaults to 5.
            threshold (float, optional): Timing deviation threshold for finding the neighbor. Defaults to 0.05.
        
        Returns: (n_prev_neighbor, n_next_neighbor)
        """

        prev_neighbors = self.na[np.abs(self.na['offset_sec'] - note['onset_sec']) < threshold]
        prev_neighbors = prev_neighbors[np.abs(prev_neighbors['pitch'] - note['pitch']) <= diameter]
        prev_neighbors = prev_neighbors[prev_neighbors['id'] != note['id']]

        next_neighbors = self.na[np.abs(self.na['onset_sec'] - note['offset_sec']) < threshold]
        next_neighbors = next_neighbors[np.abs(next_neighbors['pitch'] - note['pitch']) <= diameter]
        next_neighbors = next_neighbors[next_neighbors['id'] != note['id']]
       
        return len(prev_neighbors), len(next_neighbors)




if __name__ == '__main__':

    rc = RegionClassifier("/Users/huanzhang/01Acdemics/PhD/Research/Datasets/Burgmuller/b-04-annot.mid", burgmuller=True)