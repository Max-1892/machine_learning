import sys
import os.path
import numpy as np
from sklearn.linear_model import Perceptron
from stepwise_forward_selection import perform_SFS_feature_selection
from genetic_algorithm_feature_selection import *
from k_means import KMeans
from hac import HAC

'''This program reads in the test data and runs SFS and GA feature selection using k-means and HAC clustering'''
# Datasets to test
tests = [('data_sets/original/iris_data.txt', 3)]
         #('data_sets/original/spam_data.txt', 2)]

for test in tests:
   data_instances = []
   data_file = open(test[0])
   print "Running with %s" % test[0]
   for line in data_file:
       line_split = line.split(',')
       data_instances.append(map(float, line_split))
   data_instances = np.array(data_instances)

   # Run GA using k-means 
   hac_model = HAC(test[1])
   chosen_features = perform_GA_feature_selection(hac_model, "HAC", test[1], data_instances)
   feature_set = \
       [idx for idx in xrange(len(chosen_features[0])) if chosen_features[0][idx] == 1]
   print "Chosen features for HAC GA: %s" % str(chosen_features)
   for cluster in hac_model.get_clusters():
       print "HAC chosen cluster: %s" % str(cluster)
