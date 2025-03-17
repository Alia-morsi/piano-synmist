# Simulating Piano Performance Mistakes for Music Learning

To use this software in academic publications, please cite according to the following:

For the latest version, please write to morsi.alia(at)gmail(dot)com because the publication is under review. 

Or, if you will only use the functionality described in 
Morsi A, Zhang H, Maezawa A, Dixon S, Serra X. Simulating piano performance mistakes for music learning. Proceedings of the 21st Sound and Music Computing Conference SMC 2024; 2024 July 4-6; Porto, Portugal.
please retract to the git tag smc_2024 and cite the latter publication. 

for other uses, please see the included LICENSE file

This is the repository of SynMist, containing the synthesized mistake dataset as well as python scripts that generates the mistakes in a taxonomical way. ```simulate_mistakes.py``` and ```lowlvl.py``` contains functions regarding to the mid-level mistake scheduler and low-level deviation functions. ```region_classifier.py``` contains the simple texture \ technique region identifier. 


### Generating mistakes 
```
python simulate_mistakes.py <input_midi_folder> <output_midi_folder> <run_id>
```
This processes all midi performance files in <input_midi_folder>, applies mistakes to them, and saves the files in <run_id>/<output_midi_folder>
python simulate_mistakes.py --no_ts_annot '<path_prefix>/synthetic-mistake-study/data' repeat_test 'run11'

### Specifying the sampling probabilities
sampling_prob.csv has several mistake types and their associated probabilities. The probabilities should be interpreted as: the probability of a mistake type per detected 'texture', assuming that it has already been decided that there will be a mistake at a note that is classified as belonging to this texture. This is crucial because we do not have a 'no mistake' probability. Consistent with the above description,
 the rows should sum up to 1. (summing over the column doesn't make much sense).  
The sampling process is as follows: each note in the performance is classified as belonging to a textural region. Then, we specify a total number of mistakes that we want to be present per file (that is the current approach). Additionally, we could specify the playing attitude as something that would define the parameters of each of the functions, though currently this doesn't exist yet and those function parameters are randomly selected. 

Then, the these mistake numbers are 

The current set of values are not based on actual analysis..

index,forward_backward_insertion,mistouch,pitch_change,drag
is_double_note,0.4,0.3,0.1,0.2
is_scale_note,0.3,0.4,0.3,0
is_block_chords_note,0.2,0.1,0.4,0.3
others,0.1,0.2,0.2,0.2


## Data
The detailed analysis and annoation of individual mistakes from both datasets can be found [here](https://docs.google.com/spreadsheets/d/1QzKa0k5GlVt60PsUCvdDk8LiBAyWKOlPuEQl1Yf1ujA/edit#gid=0). 

### Burgmuller Dataset
For accessing the burgmuller dataset, please refering to the original paper [project page](https://sites.google.com/view/ismir2023-conspicuous-error) and find the download link. 

### Expert Novice Dataset

For the augmented version of expert-novice dataset (containing transcribed MIDIs and error-annotation), please refer to the [repository](https://github.com/anusfoil/EN-augmented-data). 


## Teacher's feedback

For interviews with piano teachers, we have put the evaluated samples into an [online questionnaire](https://golisten.ucd.ie/task/acr-test/65e4d319552c347eae0081dd). Their anonymized comments and rating are shown [here](https://drive.google.com/file/d/1YDfxdbq4xlRTDwVIJQ4Yq2oC0ev7b3th/view?usp=sharing). 


## SynMist dataset
<!-- The synthetic mistake MIDI dataset can be found in ```SynMist``` folder.  -->

![plot](asset/symist_statistics.png)

## Modifying datasets with time-based labels and obtaining new gt.

Following the examples to adapt ASAP and AMAPS, create a similar one for whichever dataset as long as it is possible to extract a 1d array of timevalues representing the locations of the annotations.

