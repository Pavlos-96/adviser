# Readme.md
## Directories
#### kp/ 
This is our main directory. All scripts must be run from here.
#### kp/results/
All outputs created by the scripts get saved here.
#### kp/results/our_results/
This contains the results we got by running the codes.
 comparable.
#### kp/report/
This contains everything used for the report.
#### kp/report/literature/
This contains literature, we use in the report.
#### kp/report/files_for_graphs/
This contains the files we used to construct the plots in the report.
#### kp/data/
This contains the data we used.
#### kp/src/
This contains all scripts.

## Scripts

### `creating_baselines.py`
##### Requirements
NLTK and Wordnet must be installed. 

##### Usage
```sh
$ python3 src/creating_baselines.py
```
### `Multiclass_Perceptron_domain_tracker.py`
##### Requirements
NLTK must be installed. 
##### Usage
```sh
$ python3 src/Multiclass_Perceptron_domain_tracker.py FS1
$ python3 src/Multiclass_Perceptron_domain_tracker.py FS2
```
Takes the argument "FS1" if the first feature set should be used and the argument "FS2" if the second feature set should be used.
### `train_test_split.py`
Creates a new train-test split (80/20). Using this script overwrites the train and test files we used to obtain our results.
##### Usage
```sh
$ python3 src/train_test_split.py
```

### `domain_tracker.py`
This file was copied from the adviser and modified for our purposes. The script is necessary for 'creating_baselines.py' to work and should not be run directly.
