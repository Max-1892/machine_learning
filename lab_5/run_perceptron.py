import numpy as np
import pdb
from perceptron import Perceptron

# Data sets
#tests = ['data_sets/encoded/cancer_data_encoded_NB_log.txt',
#         'data_sets/encoded/iris_data_encoded_normalized_1.txt',
#         'data_sets/encoded/iris_data_encoded_normalized_2.txt',
#         'data_sets/encoded/iris_data_encoded_normalized_3.txt',
#         'data_sets/encoded/soybean_data_encoded_NB_1.txt',
#         'data_sets/encoded/soybean_data_encoded_NB_2.txt',
#         'data_sets/encoded/soybean_data_encoded_NB_3.txt',
#         'data_sets/encoded/soybean_data_encoded_NB_4.txt',
#         'data_sets/encoded/vote_data_encoded_NB_log.txt']
tests= ['data_sets/encoded/glass_data_1_encoded.txt',
        'data_sets/encoded/glass_data_2_encoded.txt',
        'data_sets/encoded/glass_data_3_encoded.txt',
        'data_sets/encoded/glass_data_4_encoded.txt',
        'data_sets/encoded/glass_data_5_encoded.txt',
        'data_sets/encoded/glass_data_6_encoded.txt',
        'data_sets/encoded/glass_data_7_encoded.txt']

bias = {
         'data_sets/encoded/cancer_data_encoded_NB_log.txt': 20.0,
         'data_sets/encoded/iris_data_encoded_normalized_1.txt': 20.0,
         'data_sets/encoded/iris_data_encoded_normalized_2.txt': 20.0,
         'data_sets/encoded/iris_data_encoded_normalized_3.txt': 20.0,
         'data_sets/encoded/soybean_data_encoded_NB_1.txt': 20.0,
         'data_sets/encoded/soybean_data_encoded_NB_2.txt': 20.0,
         'data_sets/encoded/soybean_data_encoded_NB_3.txt': 20.0,
         'data_sets/encoded/soybean_data_encoded_NB_4.txt': 0.0,
         'data_sets/encoded/vote_data_encoded_NB_log.txt': 20.0,
         'data_sets/encoded/glass_data_1_encoded.txt': 20.0,
         'data_sets/encoded/glass_data_2_encoded.txt': 20.0,
         'data_sets/encoded/glass_data_3_encoded.txt': 20.0,
         'data_sets/encoded/glass_data_4_encoded.txt': 20.0,
         'data_sets/encoded/glass_data_5_encoded.txt': 20.0,
         'data_sets/encoded/glass_data_6_encoded.txt': 20.0,
         'data_sets/encoded/glass_data_7_encoded.txt': 20.0
       }

for test in tests:
    data_instances = []
    data_file = open(test)
    print "Running with %s" % test
    for line in data_file:
        # Digest read data
        line_split = line.split(',')
        data_instances.append(map(float, line_split))
    data_instances = np.array(data_instances)
    # Normalize continuous attributes
    if 'iris' in test:
        for column in data_instances.T:
            column = (column - np.mean(column)) / (2.0 * np.std(column))
    # Shuffle data instances
    np.random.shuffle(data_instances)

    data_indices = [idx for idx in xrange(data_instances.shape[0])]
    # 10-fold cross validation
    fold_size = (data_instances.shape[0]) / 10
    total_performance = 0.0
    for holdout_fold_idx in xrange(10):
        print "Cross validation fold %d" % (holdout_fold_idx + 1)
        # training_indices = data_indices - holdout_fold indices
        training_indices = np.array( 
            np.setdiff1d(
                data_indices, 
                data_indices[fold_size * holdout_fold_idx : \
                             fold_size * holdout_fold_idx + fold_size]))
        # test_indices = holdout_fold indices
        test_indices = np.array([i for i in xrange(
            fold_size * holdout_fold_idx, fold_size * holdout_fold_idx + fold_size)])

        model = Perceptron(bias[test])
        # Train the model
        model.train(data_instances[training_indices])
        # Test performance on test set
        predictions = model.predict(data_instances[test_indices, :-1])
        total_performance += \
            sum(predictions == data_instances[test_indices, -1]) / \
            float(test_indices.shape[0])
    print "Average overall classification rate: %f" % (total_performance / 10)
