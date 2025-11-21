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
        np.random.seed(None)  
        layer_sizes = [self.num_features] + self.neurons_per_layer + [self.num_classes]
        sz = len(layer_sizes)-1
        for i in range(sz):
            limit = np.sqrt(6.0 / (layer_sizes[i] + layer_sizes[i+1]))
            weight = np.random.uniform(-limit, limit, (layer_sizes[i], layer_sizes[i+1]))
            self.weights.append(weight)
            if self.use_bias:
                bias = np.zeros((1, layer_sizes[i+1]))
                self.biases.append(bias)
            else:
                self.biases.append(np.zeros((1, layer_sizes[i+1])))
    
    def activation(self, x):
        if self.activation_function == "sigmoid":
            # x = np.clip(x, -500, 500)
            return 1 / (1 + np.exp(-x))
        elif self.activation_function == "tanh" or self.activation_function == "hyperbolic tangent":
            # x = np.clip(x, -50, 50)
            return np.tanh(x)
        else:
            return x
    
    def derivative(self, x):
        if self.activation_function == "sigmoid":
            return x*(1-x)
        elif self.activation_function == "tanh" or self.activation_function == "hyperbolic tangent":
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
        deltas[-1] = (activations[-1]-y_true) * self.derivative(activations[-1])
        sz = len(self.weights)
        for i in range(sz-2, -1, -1):
            error = np.dot(deltas[i+1], self.weights[i+1].T)
            deltas[i] = error * self.derivative(activations[i+1])
            
        # update weights
        # new = old + (lr * delta * input)
        for i in range(sz):
            weight_gradient = np.dot(activations[i].T, deltas[i])
            
            # clip to prevent explosion
            # weight_gradient = np.clip(weight_gradient, -10, 10)
            
            # update
            self.weights[i] -= self.learning_rate * weight_gradient
            if self.use_bias:
                # bias_gradient = np.mean(deltas[i], axis=0, keepdims=True)
                bias_gradient = np.sum(deltas[i], axis=0, keepdims=True)
                self.biases[i] -= self.learning_rate * bias_gradient

    def train(self, X_train, y_train):
        print(f"class distribution: {np.sum(y_train, axis=0)}")
        
        for epoch in range(self.n_epochs):
            # indexes = np.arange(len(X_train))
            # np.random.shuffle(indexes)
            # X_shuffled = X_train[indexes]
            # y_shuffled = y_train[indexes]
            
            epoch_loss = 0
            sz = len(X_train)
            
            for i in range(sz):
                x_sample = X_train[i].reshape(1, -1)
                y_sample = y_train[i].reshape(1, -1)
                
                # forward pass
                activations = self.forward_propagation(x_sample)
                
                # loss
                epoch_loss += np.sum((y_sample - activations[-1]) ** (1<<1)) / (1<<1)
                
                # backward pass and update weights
                self.backward_propagation(activations, y_sample)
            
            self.errors_history.append(epoch_loss / len(X_train))
            
            if epoch % 200 == 0 or epoch == self.n_epochs - 1:
                # calc training
                all_activations = self.forward_propagation(X_train)
                acc = np.mean(np.argmax(y_train, axis=1) == np.argmax(all_activations[-1], axis=1)) * 100
                print(f"Epoch {epoch:4d} | Loss: {self.errors_history[-1]:.6f} | Acc: {acc:.2f}%")

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
    