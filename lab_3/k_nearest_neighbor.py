import numpy as np
from math import sqrt

class KNearestNeighbor:
    def __init__(self, k, learner_type):
        '''Class constructor'''
        self.k = k
        self.learner_type = learner_type

    def train(self, training_data):
        '''k-NN is a lazy learner so this method just stores the training data'''
        self.training_data = training_data

    def predict(self, test_data):
        '''This method tries to predict the label values of the inputted data'''
        predictions = []
        for test_instance in test_data:
            neighbors = []
            for training_instance in training_data:
                distance = sqrt(sum((test_instance - training_instance[:,:-1])**2)))
                neighbors.append((distance, training_instance))
            neighbors = sorted(neighbors, key=lambda tup: tup[0])[:self.k]
            prediction = 0.0
            for neighbor in neighbors:
                prediction += neighbor[1][-1]
            prediction = prediction / len(neighbors)
            if self.learner_type == "CLASSIFICATION":
                prediction = int(round(prediction))
            predictions.append(prediction)
