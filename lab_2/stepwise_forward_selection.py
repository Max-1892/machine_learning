from sklearn.linear_model import Perceptron
import pdb;
def perform_SFS_feature_selection(model, training_set, test_set):
    feature_set = [i for i in xrange(training_set.shape[1] - 1)]
    chosen_features = []
    base_performance = float("-inf")
    while len(feature_set) > 0:
        best_performance = float("-inf")
        #print "best performance = %f" % best_performance
        for feature in feature_set:
            chosen_features.append(feature)
            # Train model
            model = Perceptron()
            #print "Modeling with %s" % chosen_features
            model.fit(training_set[:, chosen_features], training_set[:,-1])
            # Check performance on test set
            predictions = model.predict(test_set[:, chosen_features])
            current_performance = float((predictions == test_set[:,-1]).sum()) / test_set.shape[0]
            #print "model performance = %f" % current_performance
            if current_performance > best_performance:
                best_performance = current_performance
                best_feature = feature
                #print "best performance updated to %f" % best_performance
            chosen_features.remove(feature)
        if best_performance > base_performance:
            base_performance = best_performance
            feature_set.remove(best_feature)
            chosen_features.append(best_feature)
            #print "base performance = %f" % base_performance
        else:
            print "best performance = %f" % base_performance
            break
    return chosen_features
    


