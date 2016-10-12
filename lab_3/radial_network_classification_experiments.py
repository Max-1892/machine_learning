from k_means import KMeans
from radial_basis_network import RadialBasisNetwork
import pandas as pd
import numpy as np
import pdb

tests = [('data_sets/originals/image_data_1.txt', 2), \
         ('data_sets/originals/image_data_2.txt', 2), \
         ('data_sets/originals/image_data_3.txt', 2), \
         ('data_sets/originals/image_data_4.txt', 2), \
         ('data_sets/originals/image_data_5.txt', 2), \
         ('data_sets/originals/image_data_6.txt', 2), \
         ('data_sets/originals/image_data_7.txt', 2), \
         ('data_sets/originals/ecoli_data_1.txt', 2), \
         ('data_sets/originals/ecoli_data_2.txt', 2), \
         ('data_sets/originals/ecoli_data_3.txt', 2), \
         ('data_sets/originals/ecoli_data_4.txt', 2), \
         ('data_sets/originals/ecoli_data_5.txt', 2), \
         ('data_sets/originals/ecoli_data_6.txt', 2), \
         ('data_sets/originals/ecoli_data_7.txt', 2), \
         ('data_sets/originals/ecoli_data_8.txt', 2)]

best_num_of_means = {
         'data_sets/originals/image_data_1.txt': 5,
         'data_sets/originals/image_data_2.txt': 41,
         'data_sets/originals/image_data_3.txt': 3,
         'data_sets/originals/image_data_4.txt': 19,
         'data_sets/originals/image_data_5.txt': 35,
         'data_sets/originals/image_data_6.txt': 21,
         'data_sets/originals/image_data_7.txt': 19,
         'data_sets/originals/ecoli_data_1.txt': 45,
         'data_sets/originals/ecoli_data_2.txt': 46,
         'data_sets/originals/ecoli_data_3.txt': 5,
         'data_sets/originals/ecoli_data_4.txt': 5,
         'data_sets/originals/ecoli_data_5.txt': 5,
         'data_sets/originals/ecoli_data_6.txt': 5,
         'data_sets/originals/ecoli_data_7.txt': 15,
         'data_sets/originals/ecoli_data_8.txt': 49}

# Read in data
for test in tests:
    data_instances = []
    data_file = open(test[0])
    print "Running with %s" % test[0]
    for line in data_file:
        line_split = line.split(',')
        data_instances.append(map(float, line_split))
    data_instances = np.array(data_instances)
    np.random.shuffle(data_instances)
    learner_type = "CLASSIFICATION"

    for do_clustering in [True, False]: 
        print "Performing k-means clustering first? %d" % do_clustering
        # 10 fold cross validation
        fold_size = data_instances.shape[0] / 10
        data_indices = [idx for idx in xrange(data_instances.shape[0])]
        total_performance = 0.0
        for holdout_fold_idx in xrange(10):
            rbn = RadialBasisNetwork(0.002, learner_type)
            # Train the network
            rbn.train(data_instances[ \
                        np.array( \
                            np.setdiff1d(data_indices, data_indices[ \
                                    fold_size * holdout_fold_idx : \
                                    fold_size * holdout_fold_idx + fold_size]))], \
                                    do_clustering, best_num_of_means[test[0]])

            # Predict test instances
            predictions = rbn.predict(
                    data_instances[np.array(
                        data_indices[
                            fold_size * holdout_fold_idx : 
                            fold_size * holdout_fold_idx + fold_size])])
            if learner_type == "REGRESSION":
                total_performance += \
                        sum((data_instances[np.array(
                            data_indices[
                                fold_size * holdout_fold_idx : 
                                fold_size * holdout_fold_idx + fold_size]),-1] - np.array(predictions)) ** 2)
            else:
                total_performance += \
                        sum(data_instances[np.array(
                            data_indices[
                                fold_size * holdout_fold_idx : 
                                fold_size * holdout_fold_idx + fold_size]),-1] == np.array(predictions)) / float(fold_size)
                    
        total_performance = total_performance / 10
        print "Ave classification success: %f\n" % total_performance
