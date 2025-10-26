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
                x_i = X_train[i]
                target = y_train[i]

                # Step 4: Evaluate the output y_i
                predicted_y = self.predict(x_i) # This is 'y_i' in the image
                
                # Step 5: Check if y_i != t_i
                if predicted_y != target:
                    
                    # --- This is the "then" block ---
                    
                    # Calculate 'loss' L = (t_i - y_i)
                    error = target - predicted_y
                    
                    # Form a new weight vector W_i+1 = W_i + n.(t_i - y_i).x_i
                    update = self.eta * error
                    self.weights = self.weights + (update * x_i)
                    if self.add_bias:
                        self.bias = self.bias + update
        
    def test(self, X_test, y_test): #returns y_pred
        pass
    
    def predict(self, sample): #sample
        signum_input = self.signum_input(sample)
        label = self.signum(signum_input)
        return label