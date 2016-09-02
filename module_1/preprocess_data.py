import numpy as np
import os

# Read in data 
my_data = np.genfromtxt('cancer_data.txt', \
    delimiter=',', dtype='int32')

# Split instances into classes
incomplete_instances = []
negative_instances = []
positive_instances = []
for row in my_data:
    if -1 in row:
        incomplete_instances.append(row)
    elif row[-1] == 0:
        negative_instances.append(row)
    elif row [-1] == 1:
        positive_instances.append(row)
incomplete_instances = np.array(incomplete_instances)
negative_instances = np.array(negative_instances)
positive_instances = np.array(positive_instances)

# Fill in missing attribute data by randomly 
# choosing an attribute value based on the
# conditional probability given the class
for idx, elem in np.ndenumerate(incomplete_instances):
    if elem == -1:
        if incomplete_instances[idx[0],-1] == 0:
            random_attr_val = np.random.choice(negative_instances[:,idx[1]])
        elif incomplete_instances[idx[0],-1] == 1:
            random_attr_val = np.random.choice(positive_instances[:,idx[1]])
        incomplete_instances[idx] = random_attr_val

np.savetxt('test.out', my_data, delimiter=',', fmt='%d')

# Convert discrete attributes into boolean ones
# For the cancer data, each attribute can
# range from 1 - 10 so we need 10 * 9 "new" boolean
# features
converted_result = ""
file_in = open('test.out', 'r')
file_out = open('cancer_data_encoded.txt', 'w')
for line in file_in:
    split_line = line.split(',')
    for attr in split_line[:-1]:
        converted_result += \
            (",".join("1" if int(attr) == i + 1 else "0" for i in range(10))) + ","
    converted_result += split_line[-1]
file_out.write(converted_result)
os.remove('test.out')

