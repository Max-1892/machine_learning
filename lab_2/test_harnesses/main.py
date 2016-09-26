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
tests = [('data_sets/original/glass_data.txt', 7)]
         #('data_sets/original/iris_data.txt', 3)]
         #('data_sets/original/spam_data.txt', 2)]

for test in tests:
   data_instances = []
   data_file = open(test[0])
   print "Running with %s" % test[0]
   for line in data_file:
       line_split = line.split(',')
       data_instances.append(map(float, line_split))
   data_instances = np.array(data_instances)

   # Run SFS using k-means and HAC
   kmeans_model = KMeans(test[1])
   hac_model = HAC(test[1])
   '''chosen_features = perform_SFS_feature_selection( \
       kmeans_model, "Kmeans", test[1], data_instances)
   print "K-means chosen features: %s" % str(chosen_features)'''
   chosen_features = perform_SFS_feature_selection( \
       hac_model, "HAC", test[1], data_instances)
   print "HAC chosen features: %s" % str(chosen_features)

   # Run GA feature selection using k-means and HAC
   '''kmeans_model = KMeans(test[1])
   hac_model = HAC(test[1])
   chosen_features = perform_GA_feature_selection(kmeans_model, "Kmeans", test[1], data_instances)
   print "Chosen features for K-means GA: %s" % str(chosen_features)
   chosen_features = perform_GA_feature_selection(hac_model, "HAC", test[1], data_instances)
   print "Chosen features for HAC GA: %s" % str(chosen_features)'''
