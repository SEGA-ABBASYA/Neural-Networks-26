import numpy as np

class PerceptronModel:
    def __init__(self, learning_rate, n_epochs, add_bias):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.add_bias = add_bias
        self.weights = None
        self.bias = None
        
    def initialize_weights (self):
        n_features = 2
        self.weights = np.zeros(n_features) 
        self.add_bias = 0.0
        
    def signum_input(self, sample):
        result = np.dot(sample, self.weights) + self.bias # result → W^T . x_i + b
        return result
    
    def signum(self, signum_input):
        if signum_input >= 0:
            return 1
        else:
            return -1

    def train(self, X_train, y_train):
        if self.weights is None:
            self.initialize_weights
        
        n_samples = len(y_train)
        epochs_counter = 0
        while epochs_counter < self.n_epochs:
            for i in range(n_samples):
                sample = X_train[i]
                target = y_train[i]
                
                predicted_y = self.predict(sample)
                if predicted_y != target:
                    error = target - predicted_y
                    update = self.learning_rate * error
                    self.weights = self.weights + (update * sample)
                    
                    if self.add_bias:
                        self.bias = self.bias + update
        epoch_counter ++                
        
    def test(self, X_test): #returns y_pred
        predicted_y = []
        for sample in X_test:
            predicted_y.append(self.predict(sample))
        return predicted_y    
    
    def predict(self, sample): #sample
        signum_input = self.signum_input(sample)
        label = self.signum(signum_input)
        return label