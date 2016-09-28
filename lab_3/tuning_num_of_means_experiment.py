from k_means import KMeans
from k_nearest_neighbor import KNearestNeighbor

tests = [('data_sets/originals/', )]

# Read in data
for test in tests:
    data_instances = []
    data_file = open(test[0])
    print "Running with %s" % test[0]
    for line in data_file:
        line_split = line.split(',')
        data_instances.append(map(float, line_split))
    data_instances = np.array(data_instances)

    # 10 fold cross validation
    fold_size = data_instances.shape[0] / 10
    data_indices = [idx for idx in xrange(data_instances.shape[0])]
    for num_of_means in xrange(50):
        total_performance = 0.0
        for holdout_fold_idx in xrange(10):
            # try some num of means
            kmeans_model = KMeans(num_of_means)
            # run k means on training data to find centroids
            kmeans_model.cluster( \
                data_instances[np.array( \
                    data_indices - data_indices[ \
                                          fold_size * holdout_fold_idx : \
                                          fold_size * holdout_fold_idx + fold_size]),:-1]
            # TODO: centroids don't have label in them!
            centroids = kmeans_model.get_centroids()
            #     for classification, vote to determine centroid classification
            #     for regression, average to find centroid estimate
            #  feed centroids into k-NN as training data
            if test[0] contains "ecoli" or test[0] contains "image":
                learner_type = "CLASSIFICATION"
            else:
                learner_type = "REGRESSION"
            kNN_model = KNearestNeighbor(5, learner_type)
            kNN_model.train(centroids)
            #  predict test data using k-NN and average performance
            kNN_model.predict( \
                data_instances[ \
                    fold_size * holdout_fold_idx : \
                    fold_size * holdout_fold_idx + fold_size])
            total_performance += performance
        ave_performance = total_performance / 10
        print "num of means = %d, score = %f" % (num_of_means, ave_performance)
