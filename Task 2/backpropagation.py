import numpy as np
import time

class BackpropagationModel:
    def __init__(self, num_features, num_classes, num_hidden_layers, neurons_per_layer, learning_rate=0.01, n_epochs=100,
                 use_bias=True, activation_function='sigmoid'):
        
        self.num_features = num_features
        self.num_classes = num_classes
        self.num_hidden_layers = num_hidden_layers
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.use_bias = use_bias
        self.activation_function = activation_function
        
        if isinstance(neurons_per_layer, int):
            self.neurons_per_layer = [neurons_per_layer]*num_hidden_layers
        else:
            self.neurons_per_layer = neurons_per_layer
            
        self.weights = []
        self.biases = []
        self.errors_history = []
        
        self.initialize_weights()
    
    def initialize_weights(self):
        layer_sizes = [self.num_features] + self.neurons_per_layer + [self.num_classes]
        sz = len(layer_sizes)-1
        for i in range(sz):
            weight = np.random.uniform(-0.5, 0.5, (layer_sizes[i], layer_sizes[i+1]))
            self.weights.append(weight)
            if self.use_bias:
                bias = np.random.uniform(-0.5, 0.5, (1, layer_sizes[i+1]))
                self.biases.append(bias)
            else:
                self.biases.append(np.zeros((1, layer_sizes[i+1])))
    
    def activation(self, x):
        if self.activation_function == "sigmoid":
            return 1 / (1 + np.exp(-x))
        elif self.activation_function == "tanh":
            return np.tanh(x)
        else:
            return x
    
    def derivative(self, x):
        if self.activation_function == "sigmoid":
            return x*(1-x)
        elif self.activation_function == "tanh":
            return 1-x**(1<<1)
        else:
            return np.ones_like(x)
    
    def forward_propagation(self, X):
        activations = [X]
        cur = X
        sz = len(self.weights)  
        for i in range(sz):
            z = np.dot(cur, self.weights[i])
            if self.use_bias:
                z += self.biases[i]
            a = self.activation(z)
            activations.append(a)
            cur = a
            
        return activations
    
    def backward_propagation(self, activations, y_true):    
        deltas = [None]*len(self.weights)
        
        # (target-output)*derivative
        deltas[-1] = (y_true-activations[-1]) * self.derivative(activations[-1])
        sz = len(self.weights)
        for i in range(sz-(1<<1), -1, -1):
            error = np.dot(deltas[i+1], self.weights[i+1].T)
            deltas[i] = error * self.derivative(activations[i+1])
            
        # update weights
        # new = old + (lr * delta * input)
        for i in range(sz):
            weight_gradient = np.dot(activations[i].T, deltas[i])            
            # update
            self.weights[i] += self.learning_rate * weight_gradient
            if self.use_bias:
                self.biases[i] += self.learning_rate * deltas[i]

    def train(self, X_train, y_train):
        for epoch in range(self.n_epochs):
            epoch_loss = 0
            sz = len(X_train)
            for i in range(sz):
                x_sample = X_train[i].reshape(1,-1)
                y_sample = y_train[i].reshape(1,-1)
                #forward
                activations = self.forward_propagation(x_sample)
                # loss
                epoch_loss += np.sum((y_sample - activations[-1]) ** (1<<1)) / (1<<1)
                # backward and update
                self.backward_propagation(activations, y_sample)
            self.errors_history.append(epoch_loss/len(X_train))

    def test(self, X_test):
        if X_test.ndim == 1:
            X_test = X_test.reshape(1,-1)
        activations = self.forward_propagation(X_test)
        return activations[-1]

    def predict(self, X):
        if X.ndim == 1:
            X = X.reshape(1,-1)
        activations = self.forward_propagation(X)
        return np.argmax(activations[-1], axis=1)