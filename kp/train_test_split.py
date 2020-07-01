import ast
import random

def get_data(file):
    f = open(file, "r")
    contents = f.read()
    dictionary = ast.literal_eval(contents)
    f.close()
    return dictionary

data = get_data("clean_domain_data.json")
keys = random.sample(data.keys(), len(data.keys()))
test = {}

for i in range(len(keys)//100*20):
    test[keys[i]] = data[keys[i]]
    data.pop(keys[i])
training = data


with open('train.txt', 'w') as file:
     file.write(str(training))

with open('test.txt', 'w') as file:
    file.write(str(test))