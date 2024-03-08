# Simulating and Validating Piano Performance Mistakes for Music Learning Contexts

This is the repository of SynMist, containing the synthesized mistake dataset as well as python scripts that generates the mistakes in a taxonomical way. ```mistake_by_type.py``` and ```lowlvl.py``` contains functions regarding to the mid-level mistake scheduler and low-level deviation functions. ```region_classifier.py``` contains the simple texture \ technique region identifier. 


### Generating mistakes 
```
python mistake_by_type.py --path /path/to/your/file.mid
```


## Data
The detailed analysis and annoation of individual mistakes from both datasets can be found [here](https://docs.google.com/spreadsheets/d/1QzKa0k5GlVt60PsUCvdDk8LiBAyWKOlPuEQl1Yf1ujA/edit#gid=0). 

### Burgmuller Dataset
For accessing the burgmuller dataset, please refering to the original paper [project page](https://sites.google.com/view/ismir2023-conspicuous-error) and find the download link. 

### Expert Novice Dataset

For the augmented version of expert-novice dataset (containing transcribed MIDIs and error-annotation), please refer to the [repository](https://github.com/anusfoil/EN-augmented-data). 