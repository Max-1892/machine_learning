import numpy as np
import pdb
from radial_basis_network import RadialBasisNetwork

# Data sets
tests = ['data_sets/cancer_data.txt',
     'data_sets/glass_data.txt',
     'data_sets/iris_data.txt',
     'data_sets/soybean_data.txt',
     'data_sets/vote_data.txt']

num_of_outputs = {
     'data_sets/cancer_data.txt': 2,
     'data_sets/glass_data.txt': 7,
     'data_sets/iris_data.txt': 3,
     'data_sets/soybean_data.txt': 4,
     'data_sets/vote_data.txt':  2,
}

learning_rates = {
    'data_sets/cancer_data.txt': 0.2,
    'data_sets/glass_data.txt': 0.2,
    'data_sets/iris_data.txt': 0.02,
    'data_sets/soybean_data.txt': 0.2,
    'data_sets/vote_data.txt':  0.00002,
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
        for colm_idx in xrange(data_instances.shape[1] - 1):
            column = data_instances[:, colm_idx]
            data_instances[:, colm_idx] = (column - np.mean(column)) / (2.0 * np.std(column))
    # Shuffle data instances
    np.random.shuffle(data_instances)

    #for spread in [np.random.randint(0, 100000) for i in xrange(10)]:
    #for spread in [i for i in xrange(1, 500, 7)]:
    for spread in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
        print "Testing with spread = %f" % spread
        data_indices = [idx for idx in xrange(data_instances.shape[0])]
        # 5-fold cross validation
        num_of_folds = 5
        fold_size = (data_instances.shape[0]) / num_of_folds
        total_performance = 0.0
        for holdout_fold_idx in xrange(num_of_folds):
            # training_indices = data_indices - holdout_fold indices
            training_indices = np.array( 
                np.setdiff1d(
                    data_indices, 
                    data_indices[fold_size * holdout_fold_idx : \
                                 fold_size * holdout_fold_idx + fold_size]))
            # test_indices = holdout_fold indices
            test_indices = np.array([i for i in xrange(
                fold_size * holdout_fold_idx, fold_size * holdout_fold_idx + fold_size)])
    
            model = RadialBasisNetwork(learning_rates[test], spread, num_of_outputs[test])
            # Train the model
            model.train(data_instances[training_indices])
            # Test performance on test set
            predictions = model.predict(data_instances[test_indices, :-1])
            total_performance += \
                sum(predictions == data_instances[test_indices, -1]) / \
                float(test_indices.shape[0])
        print "Average overall classification rate: %f" % (total_performance / num_of_folds)
