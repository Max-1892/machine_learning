import numpy as np
import pdb
from regression_decision_tree import RegressionDecisionTree

tests = ['data_sets/cpu_data.txt']
#tests = ['data_sets/cpu_data.txt', 
          #'data_sets/fire_data.txt', 
          #'data_sets/red_wine_data.txt', 
          #'data_sets/white_wine_data.txt']

attr_info_map = {
                 'data_sets/cpu_data.txt':
                     [("CONTINUOUS", []),
                      ("CONTINUOUS", []),
                      ("CONTINUOUS", []),
                      ("CONTINUOUS", []),
                      ("CONTINUOUS", []),
                      ("CONTINUOUS", []),
                      ("CONTINUOUS", []),
                      ("CONTINUOUS", [])],
                 'data_sets/fire_data.txt':
                     [("CONTINUOUS", []),
                      ("CONTINUOUS", []),
                      ("CONTINUOUS", []),
                      ("CONTINUOUS", []),
                      ("CONTINUOUS", []),
                      ("CONTINUOUS", []),
                      ("CONTINUOUS", []),
                      ("CONTINUOUS", []),
                      ("CONTINUOUS", []),
                      ("CONTINUOUS", []),
                      ("CONTINUOUS", [])],
                 'data_sets/white_wine_data.txt':
                     [("CONTINUOUS", []),
                      ("CONTINUOUS", []),
                      ("CONTINUOUS", []),
                      ("CONTINUOUS", []),
                      ("CONTINUOUS", []),
                      ("CONTINUOUS", []),
                      ("CONTINUOUS", []),
                      ("CONTINUOUS", []),
                      ("CONTINUOUS", []),
                      ("CONTINUOUS", []),
                      ("CONTINUOUS", [])],
                 'data_sets/red_wine_data.txt':
                     [("CONTINUOUS", []),
                      ("CONTINUOUS", []),
                      ("CONTINUOUS", []),
                      ("CONTINUOUS", []),
                      ("CONTINUOUS", []),
                      ("CONTINUOUS", []),
                      ("CONTINUOUS", []),
                      ("CONTINUOUS", []),
                      ("CONTINUOUS", []),
                      ("CONTINUOUS", []),
                      ("CONTINUOUS", [])],
                }

# Read in data
for test in tests:
    data_instances = []
    data_file = open(test)
    print "Running with %s" % test
    for line in data_file:
        line_split = line.split(',')
        data_instances.append(map(float, line_split))
    data_instances = np.array(data_instances)
    np.random.shuffle(data_instances)

    # Construct validation set
    data_indices = [idx for idx in xrange(data_instances.shape[0])]
    validation_indices = data_indices[:int(data_instances.shape[0] * 0.10)]
    validation_instances = data_instances[validation_indices]
    # Remove validation instances from data set
    data_instances = data_instances[np.setdiff1d(data_indices, validation_indices)]
    data_indices = [idx for idx in xrange(data_instances.shape[0])]
    # 5-fold cross validation
    fold_size = (data_instances.shape[0]) / 5
    total_performance = 0.0
    for holdout_fold_idx in xrange(5):
        training_indices = np.array(
            np.setdiff1d(
                data_indices, 
                data_indices[fold_size * holdout_fold_idx : fold_size * holdout_fold_idx + fold_size]))
        test_indices = np.array([i for i in xrange(
            fold_size * holdout_fold_idx, fold_size * holdout_fold_idx + fold_size)])
        tree = RegressionDecisionTree(
            attr_info = attr_info_map[test], stopping_threshold = 0.0)
        '''TODO pruned_tree = RegressionDecisionTree(
            attr_info = attr_info_map[test], stopping_threshold = 0.0)'''
        tree.train(training_data = data_instances[training_indices],
                   attributes = [i for i in xrange(data_instances[:,:-1].shape[1])])
        predictions = tree.predict(data_instances[test_indices, :-1])
        total_performance += \
            (sum(data_instances[test_indices,-1] - predictions) ** 2) / \
                float(test_indices.shape[0])
        print (sum(data_instances[test_indices,-1] - predictions) ** 2) / \
                float(test_indices.shape[0])
    print "Average mean squared error: %f" % (total_performance / 5)
