import numpy as np
import pdb
from logistic_regression import LogisticRegression

# Data sets
#tests = [('data_sets/encoded/cancer_data_encoded_NB_log.txt', [0,1]),
#         ('data_sets/encoded/iris_data_encoded_log.txt', [1,2,3]),
#         ('data_sets/encoded/soybean_data.txt', [1,2,3,4]),
#         ('data_sets/encoded/vote_data_encoded_NB_log.txt', [0,1])]
tests = [('data_sets/encoded/glass_log.txt', [1, 2, 3, 4, 5, 6, 7])]

learning_rates = {
         'data_sets/encoded/cancer_data_encoded_NB_log.txt': 0.00000001,
         'data_sets/encoded/iris_data_encoded_log.txt': 0.000001,
         'data_sets/encoded/soybean_data.txt': 0.000001,
         'data_sets/encoded/vote_data_encoded_NB_log.txt': 0.00000001,
         'data_sets/encoded/glass_log.txt': 0.000001
    }

for test in tests:
    data_instances = []
    data_file = open(test[0])
    print "Running with %s" % test[0]
    for line in data_file:
        # Digest read data
        line_split = line.split(',')
        data_instances.append(map(float, line_split))
    data_instances = np.array(data_instances)
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

        model = LogisticRegression(learning_rates[test[0]], test[1])
        # Train the model
        model.train(data_instances[training_indices], data_instances.shape[1] - 1)
        # Test performance on test set
        predictions = model.predict(data_instances[test_indices, :-1])
        total_performance += \
            sum(predictions == data_instances[test_indices, -1]) / \
            float(test_indices.shape[0])
    print "Average overall classification rate: %f" % (total_performance / 10)
