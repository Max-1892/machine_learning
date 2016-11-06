import numpy as np
import pdb
from naive_bayes_model import NaiveBayes

# Data sets
#tests = [('data_sets/encoded/cancer_data_encoded_NB_log.txt', [0,1]),
#         ('data_sets/encoded/iris_data_encoded_NB_1.txt', [0,1]),
#         ('data_sets/encoded/iris_data_encoded_NB_2.txt', [0,1]),
#         ('data_sets/encoded/iris_data_encoded_NB_3.txt', [0,1]),
#         ('data_sets/encoded/soybean_data_encoded_NB_1.txt', [0,1]),
#         ('data_sets/encoded/soybean_data_encoded_NB_2.txt', [0,1]),
#         ('data_sets/encoded/soybean_data_encoded_NB_3.txt', [0,1]),
#         ('data_sets/encoded/soybean_data_encoded_NB_4.txt', [0,1]),
#         ('data_sets/encoded/vote_data_encoded_NB_log.txt', [0,1])]

tests= [('data_sets/encoded/glass_data_1_encoded.txt', [0,1]),
        ('data_sets/encoded/glass_data_2_encoded.txt', [0,1]),
        ('data_sets/encoded/glass_data_3_encoded.txt', [0,1]),
        ('data_sets/encoded/glass_data_4_encoded.txt', [0,1]),
        ('data_sets/encoded/glass_data_5_encoded.txt', [0,1]),
        ('data_sets/encoded/glass_data_6_encoded.txt', [0,1]),
        ('data_sets/encoded/glass_data_7_encoded.txt', [0,1])]

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
        # training_indices = data_indices - holdout_fold indices
        training_indices = np.array( 
            np.setdiff1d(
                data_indices, 
                data_indices[fold_size * holdout_fold_idx : \
                             fold_size * holdout_fold_idx + fold_size]))
        # test_indices = holdout_fold indices
        test_indices = np.array([i for i in xrange(
            fold_size * holdout_fold_idx, fold_size * holdout_fold_idx + fold_size)])

        model = NaiveBayes()
        # Train the model
        model.build_model(data_instances[training_indices])
        
        # Print model
        print "Bayes model = \n%s\n" % model.print_model()

        # Test performance on test set
        predictions = []
        for instance in data_instances[test_indices, :-1]:
            predictions.append(model.predict(instance))
        total_performance += \
            sum(predictions == data_instances[test_indices, -1]) / \
            float(test_indices.shape[0])
    print "Average classification rate: %f" % (total_performance / 10)
