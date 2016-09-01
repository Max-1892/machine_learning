class WinnowModel:
    """This class encapsulates the winnow-2 algorithm."""
    weights = []
    alpha = 2
    theta = len(weights) / 2

    def __init__(self, alpha, theta, num_of_weights, weight_initial_val):
        self.alpha = alpha
        self.theta = theta
        for idx in range(num_of_weights):
            self.weights.append(float(weight_initial_val))

    def learn(self, data_instance, label):
        weight_instance_sum = 0
        for idx, val in enumerate(self.weights):
            weight_instance_sum += self.weights[idx] * data_instance[idx]
        prediction = 1 if weight_instance_sum > self.theta else 0
        if prediction == 1 and label == 0:
            self.demote_weights(data_instance)
        elif prediction == 0 and label == 1:
            self.promote_weights(data_instance)

    def promote_weights(self, data_instance):
        for idx, val in enumerate(data_instance):
            if val == 1:
                self.weights[idx] = self.weights[idx] * self.alpha

    def demote_weights(self, data_instance):
        for idx, val in enumerate(data_instance):
            if val == 1:
                self.weights[idx] = self.weights[idx] / self.alpha

    def output_model(self):
        print self.weights
        return "Weights: [%s], alpha: %f, theta: %f" % \
            (",".join([str(weight) for weight in self.weights]), \
            self.alpha , self.theta)
