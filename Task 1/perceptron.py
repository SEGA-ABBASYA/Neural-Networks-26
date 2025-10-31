import numpy as np

class PerceptronModel:
    def __init__(self, learning_rate, n_epochs, add_bias, min_value, max_value):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.add_bias = add_bias
        self.weights = None
        self.bias = 0.0
        self.min_value = min_value
        self.max_value = max_value
        
    def initialize_weights_bias(self): 
        # weights & bias using random values within min/max range 
        # of data after being scaled with standardScaler
        
        n_features = 2
        self.weights = np.random.uniform(self.min_value, self.max_value, n_features)
        if self.add_bias:
            self.bias = np.random.uniform(self.min_value, self.max_value)
        else:
            self.bias = 0.0
        
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
        if self.weights is None:
            self.initialize_weights_bias()
        
        n_samples = len(y_train)
        epochs_counter = 0
        while epochs_counter < self.n_epochs:
            misclassifications = 0  # Track errors in this epoch
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            for i in indices:
                sample = X_train[i]
                target = y_train[i]
                
                predicted_y = self.predict(sample)
                if predicted_y != target:
                    misclassifications += 1
                    error = target - predicted_y
                    update = self.learning_rate * error
                    self.weights = self.weights + (update * sample)
                    
                    if self.add_bias:
                        self.bias = self.bias + update
            
            epochs_counter += 1            
            # Early stopping: converge when no misclassifications occur
            if misclassifications == 0:
                print(f"Converged after {epochs_counter} epochs with 0 misclassifications")
                break
        
    def test(self, X_test): #returns y_pred
        predicted_y = []
        for sample in X_test:
            predicted_y.append(self.predict(sample))
        return predicted_y    
    
    def predict(self, sample): #sample
        signum_input = self.signum_input(sample)
        label = self.signum(signum_input)
        return label
    
    def get_weights(self):
        return self.weights, self.bias