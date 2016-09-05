from winnow_model import *
import sys
import os.path
import numpy as np

if len(sys.argv) != 3:
    print "Wrong number of args: data_file num_attrs"
    sys.argv.append('test')

if not os.path.isfile(sys.argv[1]):
    print "Data file doesn't exist"
    sys.exit()

try:
    num_of_attrs = int(sys.argv[2])
except ValueError:
    print "num_attrs must be an int"
    sys.exit()

model = WinnowModel(2, 0.5, num_of_attrs, 1) # alpha, theta, number of weights, initial val of weights


# Read in processed data
data_instances = []
data_file = open(sys.argv[1], 'r')
for line in data_file:
    line_split = line.split(',')
    data_instances.append(map(int, line_split))

# 10-fold cross-validation
ave_success_rate = 0.0
split_data_instances = np.array_split(data_instances, 10)
for fold in range(0, 10):
    training_data = np.concatenate((np.delete(split_data_instances, fold, 0)), 0)
    validation_data = split_data_instances[fold]
    # Train the model
    for instance in training_data:
        model.learn(instance[:-1], instance[-1])
    # Test model
    trials = len(validation_data)
    successes = 0
    failures = 0
    for instance in validation_data:
        if model.predict(instance[:-1]) != instance[-1]:
            failures += 1
        else:
            successes += 1

    ave_success_rate += successes/float(trials)
print "Ave success rate = %f" % (ave_success_rate/10)
#print model.output_model()
