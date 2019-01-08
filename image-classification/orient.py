#!/usr/bin/python2

#############################
# **REPORT**
#
# 1.  
#       We formulate the task as a classification problem of images into different 
#   orientations based on given features of 8x8 pixels, each of 3 colors, red, 
#   green, and blue. Given a data set, we use 3 different hypotheses, i.e.
#   K-Nearest Neighbor, AdaBoost, and Neural Network, to train the classification
#   function. To put it formally, we want to train the following function
#
#       f(r11, g11, b11, r12, ..., b88) \in {0, 90, 180, 270}
#
#   such that f minimizes incorrectness in the training set.
#
#
# 2.
#   K-nearest neighbor
#       The training step of this algorithm does nothing more than pooling data
#   together and save some other parameters such as the number of neighbors and
#   how much of the data to be used in the testing step.
#       In the testing step, we pick a 10% of the training dataset at random for
#   efficiency to use in classification, then for each data entry we compute the
#   Euclidean distance from the testing data entry. Finally, we look at the top K
#   data entries with the smallest distance and take the majority vote of the
#   labels to decide which label to classify the test data entry to. We use a min-
#   heap to optimize the distance ranking process by only keeping the top smallest
#   K distances at all time.
#   
#   Neural Network
#       We use a fully-connected feed-forward neural network with Xavier 
#   initialization to solve the problem. In the training step, we utilize the
#   stochastic gradient descend to learn the neural network weights and biases for
#   efficiency reason. The back-propagation continues until the mean squared errors
#   of the model does not improve more than a certain threshold for two consecutive
#   epochs or we have run the algorithm for more than 50 epochs then we terminate.
#   
#
# 3. 
#   K-nearest neighbor
#       We assume that the Euclidean distance is a good explanation of our model.
#   The problem we encounter is how slow the classification is, so we decide to pick
#   only 10% of the data.
#   
#   Neural Network
#       We decide to use Xavier initialization to initialize weights and biases 
#   because the model does not converge fast enough with any other ways. We also 
#   decide to terminate back-propagation if the improvement is too small for two 
#   consecutive epochs because we assume that it is reaching a local minima. For 
#   efficiency reason, we only let it run for no more than 50 epochs.
#
#
# 4.
#   K-nearest neighbor
#       We parameterize K with the following values and we get the following 
#   correctness from the training data:
#       K           Correctness
#       5           66.065747614
#       7           65.5355249205
#       10          66.065747614
#   
#       Since we observe the trend that the model does not improve with larger 
#   K, we decide to keep 5 
#   
#   Neural Network
#       We use different activation functions which give us the following results:
#       function    Correctness 
#       logistic    25.0
#       tanh        39.9238293759
#       relu        73.4888653234
#       So we keep relu. For other parameters, our leaning rate is 0.1 and the hidden
#   network is of size 40, 40, 40.
#
#############################

import sys 
import nnet
import nearest 

def nearest_train(train_data, model_filename):
    X = []
    Y = []
    for i in train_data.values():
        X.append(i[1])
        Y.append(i[0])
    cls = nearest.Knn() 
    cls.fit(X, Y)
    cls.save(model_filename)

def nearest_test(test_data, model_filename):
    X = []
    Y = []
    ids = []
    for key, value in test_data.iteritems():
        ids.append(key[0])
        X.append(value[1])
        Y.append(value[0])
    cls = nearest.Knn()
    cls.load(model_filename)
    results = cls.classify(X)
    for i, result in zip(ids, results):
        print i, result
    print "Correctness:", sum([int(a==b) for a, b in zip(Y, results)])/float(len(results))*100

def adaboost_train(train_data, model_filename):
    #TODO Implement here
    # This function needs to write to <model_filename>.txt to save model data
    pass

def adaboost_test(test_data, model_filename):
    #TODO Implement here
    # This function needs to print to stdout
    pass

def nnet_train(train_data, model_filename):
    X = []
    Y = []
    for i in train_data.values():
        X.append(i[1])
        Y.append(i[0])
    cls = nnet.NNetRelu()
    cls.fit(X, Y)
    cls.save(model_filename)

def nnet_test(test_data, model_filename):
    X = []
    Y = []
    ids = []
    for key, value in test_data.iteritems():
        ids.append(key[0])
        X.append(value[1])
        Y.append(value[0])
    cls = nnet.NNetRelu()
    cls.load(model_filename)
    results = cls.classify(X)
    for i, result in zip(ids, results):
        print i, result
    print "Correctness:", sum([int(a==b) for a, b in zip(Y, results)])/float(len(results))*100

# read in train data file into a dictionary
def read_train_file(train_data_filename):
    train_data = {}
    with open(train_data_filename) as f:
        for line in f:
            content = line.strip().split()
            photo_id = content[0]
            correct_orientation = int(content[1])
            features = map(int, content[2:])
            train_data[(photo_id,correct_orientation)] = (correct_orientation, features) 
    return train_data

# read in test data file into a dictionary
def read_test_file(test_data_filename):
    test_data = {}
    with open(test_data_filename) as f:
        for line in f:
            content = line.strip().split()
            photo_id = content[0]
            correct_orientation = int(content[1])
            features = map(int, content[2:])
            test_data[(photo_id, correct_orientation)] = (correct_orientation, features)
    return test_data

def print_usage():
    print "Usage: "
    print "    %s <train|test> <train_file|test_file> <model_file> <nearest|adaboost|nnet|best>" % sys.argv[0]
    sys.exit()

if len(sys.argv) < 5:
    print_usage()

task, task_filename, model_filename, model = sys.argv[1:5]

if task == "train":
    train_data = read_train_file(task_filename)
    if model == "nearest":
        nearest_train(train_data, model_filename)
    elif model == "adaboost":
        adaboost_train(train_data, model_filename)
    elif model == "nnet":
        nnet_train(train_data, model_filename)
    else:
        print_usage()
elif task == "test":
    test_data = read_test_file(task_filename)
    if model == "nearest":
        nearest_test(test_data, model_filename)
    elif model == "adaboost":
        adaboost_test(test_data, model_filename)
    elif model == "nnet":
        nnet_test(test_data, model_filename)
    elif model == "best":
        nnet_test(test_data, model_filename)
    else:
        print_usage()
else:
    print_usage()
