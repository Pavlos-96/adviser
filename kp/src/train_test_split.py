import ast
import random
from pathlib import Path
from Multiclass_Perceptron_domain_tracker import get_data


DATA_DIRECTORY = "data"

data = get_data("clean_domain_data.json")
keys = random.sample(data.keys(), len(data.keys()))
test = {}

for i in range(len(keys)//100*20):
    test[keys[i]] = data[keys[i]]
    data.pop(keys[i])
training = data

with open(str(Path(DATA_DIRECTORY, 'train.txt')), 'w') as file:
     file.write(str(training))

with open(str(Path(DATA_DIRECTORY, 'test.txt')), 'w') as file:
    file.write(str(test))