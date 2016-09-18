import sys
import os.path
import numpy as np
from sklearn.linear_model import Perceptron
from stepwise_forward_selection import perform_SFS_feature_selection

# Datasets to test
files = ['data_sets/encoded/glass_data_1_encoded.txt', \
         'data_sets/encoded/glass_data_2_encoded.txt', \
         'data_sets/encoded/glass_data_3_encoded.txt', \
         #'data_sets/encoded/glass_data_4_encoded.txt', \
         'data_sets/encoded/glass_data_5_encoded.txt', \
         'data_sets/encoded/glass_data_6_encoded.txt', \
         'data_sets/encoded/glass_data_7_encoded.txt', \
         'data_sets/encoded/iris_data_1_encoded.txt', \
         'data_sets/encoded/iris_data_2_encoded.txt', \
         'data_sets/encoded/iris_data_3_encoded.txt']

for data_file_name in files:
   # Split each line on ','
   data_instances = []
   data_file = open(data_file_name)
   print "Running with %s" % data_file_name
   for line in data_file:
       line_split = line.split(',')
       data_instances.append(map(int, line_split))

   data_instances = np.array(data_instances)
   training_indices = \
       np.array([i for i in xrange(2*len(data_instances) / 3)])
   test_indices = \
       np.array([i for i in xrange(2*(len(data_instances) / 3), len(data_instances))])
   print training_indices
   print test_indices
   model = Perceptron()
   chosen_features = perform_SFS_feature_selection( \
       model, data_instances[training_indices], data_instances[test_indices])
   chosen_features = perform_GA_feature_selection( \
       model, data_instances[training_indices], data_instances[test_indices])
   print sorted(chosen_features)
   # 10-fold cross-validation
   '''ave_success_rate = 0.0
   split_data_instances = np.array_split(data_instances, 10)
   winnow_success = []
   bayes_success = []
   for fold in range(0, 10):
       output_verbose = False
       if fold == 0:
           output_verbose = True
       training_data = np.concatenate((np.delete(split_data_instances, fold, 0)), 0)
       validation_data = split_data_instances[fold]

       # Train the model
       # alpha, theta, number of weights, initial val of weights
       winnow_model = WinnowModel(2, float(len(data_instances[0][:-1])/2), len(data_instances[0][:-1]), 1) 
       naive_bayes = NaiveBayes()
       naive_bayes.build_model(training_data)
       for instance in training_data:
           winnow_model.learn(instance[:-1], instance[-1])
       if output_verbose:
           print "Winnow model = \n%s\n" % winnow_model.print_model()
           print "Bayes model = \n%s\n" % naive_bayes.print_model()

       # Test model
       if output_verbose:
           print "Testing on test set..."
       trials = len(validation_data)
       for model in ['Winnow', 'Bayes']:
           successes = 0
           failures = 0
           for instance in validation_data:
               if output_verbose:
                   print "Data instance: %s" % instance
               if model == 'Winnow':
                   prediction = winnow_model.predict(instance[:-1])
               else:
                   prediction = naive_bayes.predict(instance[:-1])
               if output_verbose:
                   print "%s predicts %d" % (model, prediction)
               if prediction != instance[-1]:
                   failures += 1
               else:
                   successes += 1
           if model == 'Winnow':
               winnow_success.append(successes/float(trials))
           else:
               bayes_success.append(successes/float(trials))'''