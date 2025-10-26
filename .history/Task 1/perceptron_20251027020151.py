import numpy as np

class PerceptronModel:
    def __init__(self, learning_rate, n_epochs, add_bias):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.add_bias = add_bias
        self.weights = None
        self.bias = None
        
    def initialize_weights(self, n_features):
        self.weights = np.zeros(n_features) 
        self.bias = 0.0 if self.add_bias else None
        
    def signum_input(self, sample):
        result = np.dot(sample, self.weights)
        if self.add_bias:
            result += self.bias
        return result
    
    def signum(self, signum_input):
        if signum_input >= 0:
            return 1
        else:
            return -1

    def train(self, X_train, y_train):
        # Validate input dimensions
        if len(X_train) != len(y_train):
            raise ValueError("Number of samples in X_train and y_train must match")
        if not isinstance(X_train, np.ndarray):
            X_train = np.array(X_train)
        if not isinstance(y_train, np.ndarray):
            y_train = np.array(y_train)
            
        # Initialize weights if not already initialized
        if self.weights is None:
            n_features = X_train.shape[1]
            self.initialize_weights(n_features)
        
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
            epochs_counter += 1
        
    def test(self, X_test): #returns y_pred
        predicted_y = []
        for sample in X_test:
            predicted_y.append(self.predict(sample))
        return predicted_y    
    
    def predict(self, sample): #sample
        signum_input = self.signum_input(sample)
        label = self.signum(signum_input)
        return label