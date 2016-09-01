import numpy as np

# Read in data 
my_data = np.genfromtxt('breast_cancer_data.txt', \
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
