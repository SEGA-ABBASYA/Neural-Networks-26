import numpy as np

class PerceptronModel:
    def __init__(self, learning_rate, n_epochs, add_bias, min=-1, max=1):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.add_bias = add_bias
        self.weights = None
        self.bias = 0.0
        self.min = -1
        self.max = 1

    def initialize_weights(self, n_features=2):
        self.weights = np.random.uniform(self.min, self.max, n_features).astype(float)
        self.bias = 0.0 if self.add_bias else 0.0
        
    def signum_input(self, sample):
        result = np.dot(sample, self.weights)
        if self.add_bias:
            result += self.bias
        return float(result)
    
    def signum(self, value):
        return 1 if value >= 0 else -1

    def train(self, X_train, y_train):
        if self.weights is None:
            self.initialize_weights(X_train.shape[1])
        
        n_samples = len(y_train)
        for epoch in range(self.n_epochs):
            for i in range(n_samples):
                sample = np.asarray(X_train[i], dtype=float)
                target = y_train[i]
                predicted_y = self.predict(sample)
                if predicted_y != target:
                    error = target - predicted_y
                    update = self.learning_rate * error
                    self.weights += update * sample
                    if self.add_bias:
                        self.bias += update

        
    def test(self, X_test):
        return np.array([self.predict(sample) for sample in X_test])
    
    def predict(self, sample):
        signum_input_val = self.signum_input(sample)
        return self.signum(signum_input_val)
    
    def get_weights(self):
        return self.weights, self.bias
