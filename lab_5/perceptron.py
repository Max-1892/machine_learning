import numpy as np
import pdb
import random
from copy import deepcopy

class Perceptron:
    def __init__(self, bias):
        self.bias = bias

    def train(self, training_set):
        '''This method trains the weight of the perceptron using the inputted training set.'''
        # Initialize the weights
        self.weights = np.array([random.uniform(-0.01, 0.01) for i in xrange(training_set.shape[1] - 1)])
        print "Initial weights: %s" % str(self.weights)

        # While the weights haven't converged...
        converged = False
        iterations = 0
        while not converged and iterations < 10000:
            old_weights = deepcopy(self.weights)
            for instance in training_set:
                # Calculate their dot product with the weights
                output = np.dot(instance[:-1], self.weights) - self.bias
                # This if-else serves as a sigmoid activiation function,
                # where sgn(x) = 1 when output > 0 and 0 otherwise
                if output > 0:
                    prediction = 1
                else:
                    prediction = 0
                # If the predict generated from the current weights
                # doesn't match the class label, update the weights.
                if prediction != instance[-1]:
                    if prediction == 0: 
                        # If the class label is 0 but we predicted 1, 
                        # subtract the instance from the weights
                        self.weights -= instance[:-1]
                    else: 
                        # If the class label is 1 but we predicted 0, 
                        # add the instance to the weights
                        self.weights += instance[:-1]
            print sum(abs(old_weights - self.weights))
            if sum(abs(old_weights - self.weights)) < 0.00000001:
                converged = True
            iterations += 1
        print "Learned perceptron weights: %s" % str(self.weights)

    def predict(self, test_set):
        '''This method predicts the class for each example in
           the test set by calculating the dot product of a 
           test instance and the perceptron weights and passing
           the result through an activiation function.'''
        predictions = []
        # Cycle through instances
        for instance in test_set:
            # Calculate their dot product with the weights
            output = np.dot(instance, self.weights) - self.bias
            # This if-else serves as a sigmoid activiation function,
            # where sgn(x) = 1 when output > 0 and 0 otherwise
            if output > 0:
                predictions.append(1)
            else:
                predictions.append(0)
        return predictions
