import numpy as np

class PerceptronModel:
    def __init__(self, learning_rate, n_epochs, add_bias): #Constructor
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.add_bias = add_bias
        self.weights = None
        self.bias = None
        
    def initialize_weights (self): # Small random numbers between -0.01 and 0.01
        self.weights = np.zeros(2) # number of features = 2
        self.add_bias = 0.0
        
    def signum_input(self, sample):
        # input → W^T . x_i + b
        result = np.dot(sample, self.weights) + self.bias
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
        
    def test(self, X_test): #returns y_pred
        predicted_y = []
        for sample in X_test
    
    def predict(self, sample): #sample
        signum_input = self.signum_input(sample)
        label = self.signum(signum_input)
        return label