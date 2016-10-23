from __future__ import division
from math import log
from decision_tree_node import DecisionTreeNode
import pdb
from copy import deepcopy
import numpy as np

'''This class implements a regression decision tree using gain ratio.'''
class RegressionDecisionTree:
    def __init__(self, attr_info, stopping_threshold):
        # attr_info is a list of tuples where tuple[0] is the
        # attribute type ("DISCRETE", "CONTINUOUS") and tuple[1]
        # is a list of discrete values the attribute takes or
        # ranges to consider
        self.attr_info = attr_info
        self.best_split_value_cache = {}
        self.stopping_threshold = stopping_threshold

    def calculate_var_info(self, split, left_split, right_split):
        total = 0.0
        left_total = 0.0
        right_total = 0.0
        for inst_1, left_1, right_1 in zip(split, left_split, right_split):
            for inst_2, left_2, right_2 in zip(split, left_split, right_split):
                total += sum(inst_1 - inst_2) ** 2
                left_total += sum(left_1 - left_2) ** 2
                right_total += sum(right_1 - right_2) ** 2
        if split.shape[0] != 0:
            total = (total / (split.shape[0] ** 2))
        if left_split.shape[0] != 0:
            left_total = left_total / (left_split.shape[0] ** 2)
        if right_split.shape[0] != 0:
            right_total = right_total / (right_split.shape[0] ** 2)
        return total - (left_total + right_total)

    def determine_best_split_value_for_cont_variable(self, split, attr_idx):
        split_copy = np.copy(split)
        split_copy.flags.writeable = False
        hash_val = hash(split_copy.data)
        if (hash_val, attr_idx) in self.best_split_value_cache.keys():
            return self.best_split_value_cache[(hash_val, attr_idx)]

        '''
            I'm using the method outlined in Tom Michell's Machine Learning textbook.
            Specifically, sort the split on attr_idx from low to high.
            Create a list of candidate split points from the mid-points of the attribute values.
        '''
        # Pull out the attribute we care about
        subset = split[:, attr_idx]
        # Sort on attr_idx from low to high
        subset = np.sort(subset)
        previous_instance = subset[0]
        candidate_split_points = []
        for instance in subset:
            candidate_split_points.append((float(previous_instance) + instance) / 2)
            previous_instance = instance
        # Cycle through candidate split points and return the one with the high gain
        max_var_info = float("-inf")
        max_var_split_point = -1.0
        for split_point in candidate_split_points:
            var_info = self.calculate_var_info(
                split, 
                split[np.where(split[:, attr_idx] > split_point)], 
                split[np.where(split[:, attr_idx] <= split_point)])
            if var_info > max_var_info:
                max_var_info = var_info
                max_var_split_point = split_point
        self.best_split_value_cache[(hash_val, attr_idx)] = max_var_split_point
        return max_var_split_point, max_var_info

    def determine_split_attribute(self, instances, attributes):
        max_var = float("-inf")
        best_attr_idx = -1
        best_attr_split_val = -1
        # For every attribute
        for attr_idx in attributes:
            best_split_val, max_var_info = \
                self.determine_best_split_value_for_cont_variable(instances, attr_idx)
            if max_var_info > max_var:
                max_var = max_var_info
                best_attr_idx = attr_idx
                best_attr_split_val = best_split_val
        # return best attribute
        return best_attr_idx, best_attr_split_val

    def calculate_mse(self, inst):
        mean_arr = np.array([np.mean(inst[:, -1])] * inst.shape[0])
        return sum((mean_arr - inst[:, -1]) ** 2) / inst.shape[0]

    def generate_tree(self, instances, attributes):
        self.root_node = DecisionTreeNode()
        self.root_node.instances = instances
        job_queue = []
        job_queue.append((self.root_node, instances, attributes))
        while len(job_queue) > 0:
            node, inst, attrs = job_queue.pop(0)
            if len(attrs) == 0 or self.calculate_mse(inst) < self.stopping_threshold:
                node.label = np.mean(inst[:, -1])
                node.instances = inst
                continue
            else: 
                # Determine the attribute with the best gain ratio 
                split_attr, split_val = self.determine_split_attribute(inst, attrs) 
                node.split_attr = split_attr
                node.split_attr_type = "CONTINUOUS"
                node.attr_values = split_val
                attributes.remove(split_attr) 
                subset_leq = inst[np.where(inst[:,split_attr] <= node.attr_values)]
                subset_greater = inst[np.where(inst[:,split_attr] > node.attr_values)]
                if subset_leq.shape[0] > 0 and subset_greater.shape[0] > 0: 
                    child_node = DecisionTreeNode() 
                    child_node.instances = subset_leq
                    job_queue.append((child_node, subset_leq, attrs)) 
                    node.add_branch(child_node)
                    child_node2 = DecisionTreeNode() 
                    child_node2.instances = subset_greater
                    job_queue.append((child_node2, subset_greater, attrs)) 
                    node.add_branch(child_node2)
                else:
                    node.branches = []
                    node.label = np.mean(inst[:,-1])
                    node.instances = inst
        return self.root_node 

    def train(self, training_data, attributes):
        self.root_node = self.generate_tree(training_data, attributes)

    def predict(self, test_data):
        predictions = []
        for instance in test_data:
            node = self.root_node
            while len(node.branches) != 0:
                if instance[node.split_attr] <= node.attr_values:
                    node = node.branches[0]
                else:
                    node = node.branches[1]
            predictions.append(node.label)
        return predictions
