import pdb
import numpy as np

'''This class implements a classification decision tree using gain ratio.'''
class ClassificationDecisionTree:
    def __init__(self, attr_info):
        # attr_info is a list of tuples where tuple[0] is the
        # attribute type ("DISCRETE", "CONTINUOUS") and tuple[1]
        # is a list of discrete values the attribute takes or
        # ranges to consider
        self.attr_info = attr_info

    def calculate_information(self):
        print "TODO!"

    def calculate_entropy(self):
        print "TODO!"

    def calculate_intrinsic_value(self):
        print "TODO!"

    def calculate_gain(self):
        print "TODO!"

    def determine_split_attribute(self, instances):
        min_gain_ratio = float("inf")
        best_attr_idx = -1
        # For every attribute
        for attr_idx in xrange(instances[:,:-1].shape[1]):
            # If attribute is discrete (n values)...
            if attr_info[attr_idx][0] == "DISCRETE":
                for attr_val in attr_info[attr_idx][1]:
                    # Split instances based on value of attribute
                    split = instances[np.where(instances[:, attr_idx] == attr_val)]
                    # Calculate gain_ratio
                    gain_ratio = self.calculate_gain(split) / calculate_intrinsic_value(split)
                    # if cal_gain_ratio < min_gain_ratio, reset min and index
                    if gain_ratio < min_gain_ratio:
                        min_gain_ratio = gain_ratio
                        best_attr_idx = attr_idx
            else:
                # Else attribute is continuous...
                # For all possible splits (?)
                    for interval in attr_info[attr_idx][1]:
                        # Split based on a single split
                        split = instances[np.where(instances[:, attr_idx] in interval)]
                        # Calculate gain_ratio
                        gain_ratio = self.calculate_gain(split) / calculate_intrinsic_value(split)
                        # if cal_gain_ratio < min_gain_ratio, reset min and index
                        if gain_ratio < min_gain_ratio:
                            min_gain_ratio = gain_ratio
                            best_attr_idx = attr_idx
        # return best attribute
        return best_attr_idx

    def generate_tree(self, instances):
        # TODO: Is this early stopping?
        if self.calculate_entropy(instances) < something:
            # Create leaf labelled by majority class in instances
            return
        split_attr = self.determine_split_attribute(instances)
        # for each value that split_attr can take...
        for attr_val in self.attr_info[split_attr][1]:
            # Determine instances with that value and call generate_tree on them
            if self.attr_info[split_attr][0] == "DISCRETE":
                self.generate_tree(instances[np.where(instances[:, attr_idx] == attr_val)])
            else:
                # TODO: ?
                #self.generate_tree(instances[np.where(instances[:, attr_idx] > attr_val)])

    def train(self, training_data):
        generate_tree(training_data)

    def predict(self, test_data):
        print "TODO!"
