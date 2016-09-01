import numpy as np
import os

# Read in data 
my_data = np.genfromtxt('cancer_data.txt', \
    delimiter=',', dtype='int32')

# If an element can't be read in (i.e. is -1)
# randomly choose a number from the rest of attribute
# values for that column
for ind, elem in np.ndenumerate(my_data):
    if elem == -1:
        random_attr_val = -1
        while (random_attr_val == -1):
            random_attr_val = np.random.choice(my_data[:,ind[1]])
        my_data[ind] = random_attr_val

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

