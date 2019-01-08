#Team members:
# Natanee Dokmai : ndokmai
#Sai Shruthi Umakanth : saumak
#Siddharth Shankar : sidshank

#Report:

#K- Nearest Neighbour:
#Formulation of the problem:
#	->We formulated the problem, by importing the values of the training data into a hash table, 
#	  by taking photo_id as key and all other values of it as its value. 
#	->The above method is done for both train data and test data. 
#	->Then for each photo_id, the values of the pixel points are taken into consideration and made as euclidian distance
#	  using the formula:
#						dist = (b - a)^2
#	  where b -> changed pixel value
#			a -> initial pixel value
#	-> This distance and the photo_id are taken into consideration as dictionary for k-nearest value estimation. 
#		K- NEAREST NEIGHBOUR:
#			->As for this, we have taken the value of k as 7 for calculation. 
#			->So, if we consider the key values in a form of graph, the nearest 7 points are taken 
#				into consideration as neighbours.
#			->Then the orientation of the 7 points are considered and the majority orientation value is 
#				taken as the correct coordinate for the considered photo_id.
#Assumptions:
#	One assumption we made is the value of K as 7. 
#	This is to done to take a sizable photos into consideration without overfitting the data. 
#Discussions:
#	Discussions were mostly made within the team in terms of developing the model and in taking the value of k.



###
####
#######


import heapq
import numpy as np
from collections import Counter
import pickle
class Knn:
    def __init__(self, num_neighbors=10, sample_size_factor=0.1):
        self.num_neighbors = num_neighbors
        self.X = None
        self.Y = None
        self.sample_size = None
        self.sample_size_factor = sample_size_factor

    def dist(self, a, b):
        return sum((a-b)**2)

    def add_to_heap(self, q, d, y):
        if len(q)==self.num_neighbors:
            if d<-q[0][0]:
                heapq.heappushpop(q, (-d, y))
        else:
            heapq.heappush(q, (-d, y))

    def classify(self, X):
        results = []
        for x in X:
            q = [] 
            choices = np.random.choice(len(self.X), 
                    self.sample_size, replace=False)
            train_X = [ self.X[i] for i in choices ]
            train_Y = [ self.Y[i] for i in choices ]
            for train_x, y in zip(train_X, train_Y):
                d = self.dist(x, train_x)
                self.add_to_heap(q, d, y)
            votes = [i[1] for i in q]
            counts = Counter(votes)
            result = max(counts, key=counts.get)
            results += [result]
        return results

    def fit(self, X, Y): 
        self.X = np.array(X)
        self.Y = np.array(Y)
        self.sample_size = int(len(X)*self.sample_size_factor)

    def save(self, model_filename):
        with open(model_filename, "wb") as f:
            np.save(f, self.X)
            np.save(f, self.Y)
            pickle.dump(self.sample_size,f)
            pickle.dump(self.num_neighbors,f)


    def load(self, model_filename):
        with open(model_filename, "rb") as f:
            self.X = np.load(f)
            self.Y = np.load(f)
            self.sample_size = pickle.load(f)
            self.num_neighbors = pickle.load(f)
