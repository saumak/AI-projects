from abc import ABCMeta, abstractmethod
import numpy as np
import pickle
import itertools
# source: https://stackoverflow.com/questions/5434891/iterate-a-list-as-pair-current-next-in-python

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return itertools.izip(a, b)

class NNet:
    __metaclass__ = ABCMeta

    @abstractmethod
    # activation function
    def activation(self, zs):
        pass

    @abstractmethod
    # gradient of the activation function 
    def activation_gradient(self, zs):
        pass


    def __init__(self, alpha=1e-1, hidden_layer_sizes = [40]*4, max_iter=50, batch_size=100,
            threshold=1e-3):
        self.alpha = alpha
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.threshold = threshold
        self.weights = None 
        self.biases = None 
        self.coding_map = None

    def classify(self, X):
        results = []
        for x in X:
            prev_activations = x 
            for weight, bias in zip(self.weights, self.biases):
                z = np.matmul(weight, prev_activations) + bias
                a, _ = self.activation(z)
                prev_activations = a
            _, clss = max(zip(prev_activations, range(len(prev_activations))))
            for key, values in self.coding_map.iteritems():
                if values[clss]==1:
                    results += [key]
        return results

    def evaluate(self, X, Y, weights, biases):
        accu_loss = 0.
        for x,y in zip(X, Y):
            prev_activations = x 
            for weight, bias in zip(weights, biases):
                z = np.matmul(weight, prev_activations) + bias
                a, _ = self.activation(z)
                prev_activations = a
            accu_loss += sum((prev_activations - self.binary_coding(y))**2) 
        return accu_loss/(2.*len(X))

    # gradient of C with respect to the activations of the output layer
    def output_cost_gradient(self, output_activations, labels):
        return output_activations-labels

    def weight_init(self, layer_sizes):

        weights = [ np.random.normal(0, 1/float(current_size), (next_size, current_size))
                for current_size, next_size in pairwise(layer_sizes) ]
        biases = [ [0.]*next_size
                for _, next_size in pairwise(layer_sizes) ]
        return weights, biases

    def mini_batch_select(self, X, Y):
        num_inputs = len(X)
        pairs = zip(X,Y)
        np.random.shuffle(pairs)
        X = np.array([ i[0] for i in pairs ])
        Y = np.array([ i[1] for i in pairs ])
        return zip([ X[k:k+self.batch_size] for k in xrange(0, num_inputs, self.batch_size)],
                [ Y[k:k+self.batch_size] for k in xrange(0, num_inputs, self.batch_size)])

    def make_binary_coding_map(self, Y):
        classes = set(Y)
        coding_map = dict.fromkeys(classes, [])
        for i, c in enumerate(coding_map.keys()):
            tmp = [0]*len(classes)
            tmp[i] = 1
            coding_map[c] = tmp
        return coding_map

    def binary_coding(self, y):
        return self.coding_map[y]

    def fit(self, X, Y):
        X = np.array(X)
        Y = np.array(Y)
        coding_map = self.make_binary_coding_map(Y) 
        self.coding_map = coding_map

        self.batch_size = min(self.batch_size, len(X))
        input_size = len(X[0])
        num_classes = len(set(Y))
        all_layer_sizes = np.array([input_size] + self.hidden_layer_sizes + [num_classes])
        
        # initialize weights
        weights, biases = self.weight_init(all_layer_sizes)

        prev_loss = num_classes
        second_prev_loss = prev_loss + self.threshold*2

        for rnd in range(1, self.max_iter+1):
            for batch_inputs, batch_labels in self.mini_batch_select(X, Y):
                weight_descent_accu  = [ np.array([[0.]*current_size]*next_size) 
                        for current_size, next_size in pairwise(all_layer_sizes) ]
                bias_descent_accu = [ np.array([0.]*next_size)
                        for _, next_size in pairwise(all_layer_sizes) ]

                for x, y in zip(batch_inputs, batch_labels):
                    binary_coding_y = coding_map[y]
                    activations = [] 
                    activation_gradients = []
                    zs = [] 

                    # feedforward
                    prev_activations = x 
                    for weight, bias in zip(weights, biases):
                        z = np.matmul(weight, prev_activations) + bias
                        a, d_a = self.activation(z)
                        zs += [z]
                        activations += [a]
                        activation_gradients += [d_a]
                        prev_activations = a

                    # compute loss
                    gradient = self.output_cost_gradient(activations[-1], binary_coding_y)

                    # compute output errors
                    output_delta = np.multiply(gradient, activation_gradients[-1])

                    # compute backpropagation errors
                    activation_gradients = activation_gradients[:-1]
                    activations = activations[:-1]

                    backprop_deltas = []
                    prev_delta = output_delta
                    for weight, activation_gradient in \
                            zip(weights[1:][::-1], activation_gradients[::-1]):
                        delta = np.multiply(
                                np.matmul(np.transpose(weight), prev_delta),
                                activation_gradient)
                        backprop_deltas += [delta]
                        prev_delta = delta
                    backprop_deltas.reverse()
                    weight_descent_accu = [ a+np.array(
                        np.matmul(np.transpose(np.matrix(b)), np.matrix(c))) for a,b,c in 
                            zip(weight_descent_accu, backprop_deltas+[output_delta],
                                [x]+activations) ]
                    bias_descent_accu = [ a+b for a,b in 
                            zip(bias_descent_accu, backprop_deltas+[output_delta]) ]
                weight_descent = [-self.alpha/float(self.batch_size)*a for a in weight_descent_accu]
                bias_descent = [-self.alpha/float(self.batch_size)*a for a in bias_descent_accu]
                weights = [ a+b for a, b in zip(weights, weight_descent) ]
                biases = [ a+b for a, b in zip(biases, bias_descent) ]
            loss = self.evaluate(X,Y, weights, biases) 
            print "round:", rnd, "loss:", loss
            if abs(loss-prev_loss)<self.threshold and \
                    abs(prev_loss-second_prev_loss)<self.threshold:
                break
            second_prev_loss = prev_loss
            prev_loss = loss
        self.weights = weights
        self.biases = biases
        results = self.classify(X)
        corrects = sum([int(a==b) for a,b in zip(results, Y)])
        percent = corrects/float(len(results))*100

        print "Percentage of correctness:", percent

    def save(self, model_filename):
        with open(model_filename, "wb") as f:
            np.save(f, self.weights)
            np.save(f, self.biases)
            pickle.dump(self.coding_map,f)

    def load(self, model_filename):
        with open(model_filename, "rb") as f:
            self.weights = np.load(f)
            self.biases = np.load(f)
            self.coding_map = pickle.load(f) 


class NNetLogistic(NNet):

    # return both activations and gradeints for optimization
    def activation(self, zs):
        a = 1./(1. + np.exp(-zs))
        return (a, a*(1.-a))

    def activation_gradient(self, zs):
        a = 1./(1. + np.exp(-zs))
        return a*(1.-a) 


class NNetTanh(NNet):

    # return both activations and gradeints for optimization
    def activation(self, zs):
        a = np.tanh(zs) 
        return (a, 1-a**2)

    def activation_gradient(self, zs):
        a = np.tanh(zs) 
        return 1-a**2 

class NNetRelu(NNet):

    def activation(self, zs):
        return (np.clip(zs, 0, None), [ 1 if i>=0 else 0 for i in zs]) 

    def activation_gradient(self, zs):
        return [ 1 if i>=0 else 0 for i in zs]

class NNetID(NNet):

    def activation(self, zs):
        return (zs, [1.]*len(zs)) 

    def activation_gradient(self, zs):
        return [1.]*len(zs)
