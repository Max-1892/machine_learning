from k_means import KMeans;
from hac import HAC;
'''This method implements stepwise forward feature selection.'''
def perform_SFS_feature_selection(model, model_type, num_of_classes, data_set):
    # Create a boolean string, 1 = include feature, 0 = leave it out
    feature_set = [i for i in xrange(data_set.shape[1])]
    chosen_features = []
    chosen_clusters = []
    base_performance = float("-inf")
    # while there are still features to choose from...
    while len(feature_set) > 0:
        # initialize performance metrics
        best_performance = float("-inf")
        best_clusters = []
        #print "best performance = %f" % best_performance
        # Pick a feature that hasn't be chosen yet and train the model
        for feature in feature_set:
            chosen_features.append(feature)
            # Train model 
            if model_type == "Kmeans":
                model = KMeans(num_of_classes)
            elif model_type == "HAC":
                model = HAC(num_of_classes)
            #print "Modeling with %s" % chosen_features
            clusters = model.cluster(data_set)
            # Calculate performance via LDA-like objective function
            current_performance = model.calculate_performance()
            #print "model performance = %f" % current_performance
            # if this combo of features beats the best performance so far
            # take note...
            if current_performance > best_performance:
                best_performance = current_performance
                best_feature = feature
                best_clusters = clusters
                #print "best performance updated to %f" % best_performance
            chosen_features.remove(feature)
        # If best noted performance beats the best performance we've seen
        # so far, add to chosen features 
        if best_performance > base_performance:
            base_performance = best_performance
            feature_set.remove(best_feature)
            chosen_features.append(best_feature)
            chosen_clusters = best_clusters
            #print "base performance = %f" % base_performance
        else:
            #print "best performance = %f" % base_performance
            break
    return chosen_features, chosen_clusters
