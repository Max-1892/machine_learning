tests = ['data_sets/abalone_data.txt']

# Read in data
for test in tests:
    data_instances = []
    data_file = open(test[0])
    print "Running with %s" % test[0]
    for line in data_file:
        line_split = line.split(',')
        data_instances.append(map(float, line_split))
    data_instances = np.array(data_instances)
    np.random.shuffle(data_instances)


