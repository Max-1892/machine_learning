from k_means import KMeans
from radial_basis_network import RadialBasisNetwork
import pandas as pd
import numpy as np
import pdb

tests = [('data_sets/originals/cpu_data.txt', -1), \
         ('data_sets/originals/fire_data.txt', -1)]
         #('data_sets/originals/image_data_1.txt', 2), \
         #('data_sets/originals/image_data_2.txt', 2), \
         #('data_sets/originals/image_data_3.txt', 2), \
         #('data_sets/originals/image_data_4.txt', 2), \
         #('data_sets/originals/image_data_5.txt', 2), \
         #('data_sets/originals/image_data_6.txt', 2), \
         #('data_sets/originals/image_data_7.txt', 2), \
         #('data_sets/originals/ecoli_data_1.txt', 2), \
         #('data_sets/originals/ecoli_data_2.txt', 2), \
         #('data_sets/originals/ecoli_data_3.txt', 2), \
         ###('data_sets/originals/ecoli_data_4.txt', 2), \
         #('data_sets/originals/ecoli_data_5.txt', 2), \
         #('data_sets/originals/ecoli_data_6.txt', 2), \
         #('data_sets/originals/ecoli_data_7.txt', 2), \
         #('data_sets/originals/ecoli_data_8.txt', 2)]

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
    if "ecoli" in test[0] or "image" in test[0]:
        learner_type = "CLASSIFICATION"
    else:
        learner_type = "REGRESSION"

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
                                fold_size * holdout_fold_idx + fold_size]))], False)

        # Predict test instances
        predictions = rbn.predict(
                data_instances[np.array(
                    data_indices[
                        fold_size * holdout_fold_idx : 
                        fold_size * holdout_fold_idx + fold_size])])
        total_performance += \
                sum((data_instances[np.array(
                    data_indices[
                        fold_size * holdout_fold_idx : 
                        fold_size * holdout_fold_idx + fold_size]),-1] - np.array(predictions)) ** 2)
    total_performance = total_performance / 10
    print "Ave mean squared error: %f\n" % total_performance