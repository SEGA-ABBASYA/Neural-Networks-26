import numpy as np

class AdalineModel:
    """
    Adaline (Adaptive Linear Neuron)
    
    Parameters:
        learning_rate (float): Learning rate between 0.0 and 1.0
        n_epochs (int): Number of training iterations over the dataset
        mse_threshold (float): MSE threshold for early stopping (default: None)
        add_bias (bool): Whether to include bias term (default: True)
    
    Attributes:
        weights (ndarray): Feature weights after training
        bias (float): Bias term after training
        losses (list): Mean squared error values for each epoch
    """

    def __init__(self, learning_rate=1e-1, n_epochs=1e2, mse_threshold=None, add_bias=True):
        if not 0.0 < learning_rate <= 1.0:
            raise ValueError("Learning rate must be between 0.0 and 1.0")
        if n_epochs <= 0:
            raise ValueError("Number of epochs must be a positive integer")
        
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.mse_threshold = mse_threshold
        self.add_bias = add_bias
        self.weights = None
        self.bias = np.float64(0)
        self.losses = []

    def train(self, X_train, y_train):
        """
        Fit training data using gradient descent.
        
        Args:
            X_train: Training features (2D array of shape [n_samples, n_features])
            y_train: Training labels (1D array of shape [n_samples])
        
        Returns:
            self: Returns the trained model object
        """
        rand = np.random.RandomState(42)
        self.weights = rand.normal(loc=0.0, scale=0.01, size=X_train.shape[1])
        self.bias = np.float64(0.0)
        
        self.losses = []
        epoch = 0
        while epoch < self.n_epochs:
            net_input = self.calculate_weighted_sum(X_train)
            output = self.linear_activation(net_input)
            
            # Calculate errors
            errors = y_train - output
            
            # Update weights and bias using gradient descent
            self.weights += self.learning_rate * (float(1<<1) / X_train.shape[0]) * X_train.T.dot(errors)
            # self.weights += self.learning_rate*float(1<<1)*X_train.T.dot(errors)/X_train.shape[0]
            if self.add_bias:
                self.bias += self.learning_rate *float(1<<1) * errors.mean()
                # self.bias += self.learning_rate * (float(1<<1) / X_train.shape[0]) * errors.sum()
            
            # Calculate mean squared error for this epoch
            loss = np.mean(errors**2)
            self.losses.append(loss)
            # print(f"Epoch {epoch+1}/{self.n_epochs}, MSE:{loss:.6f}")
            # early stopping if the threshold is met
            if self.mse_threshold is not None and loss < self.mse_threshold:
                print(f"Early stopping at epoch {epoch+1}, MSE:{loss:.6f}")
                break

            epoch += 1

        return self

    def test(self, X_test):
        """
        Test the model.
        
        Args:
            X_test: Test features (2D array)
        Returns:
            y_pred: Predicted class labels (0 or 1)
        """
        y_pred = self.predict(X_test)
        return y_pred

    def predict(self, X):
        """
        Predict class labels after the unit step.
        
        Args:
            X: Input features (2D array)
        
        Returns:
            Predicted class labels (-1 or 1)
        """
        net_input = self.linear_activation(self.calculate_weighted_sum(X))
        prediction = np.array([1 if x >= 0 else -1 for x in net_input])
        return prediction

    def calculate_weighted_sum(self, X):
        """
        Calculate the weighted sum of inputs (net input).
        
        Args:
            X: Input features (2D array)
        
        Returns:
            Net input values (weighted sum + bias)
        """
        if self.add_bias:
            return np.dot(X,self.weights)+self.bias
        else:
            return np.dot(X,self.weights)
        

    def linear_activation(self, input):
        """
        Apply linear activation (identity function).
        
        Args:
            X: Net input values
        
        Returns:
            Same as input values cause adaline uses linear activation
        """
        return input

    def get_params(self):
        """
        Get model parameters.
        Returns
        -------
        params : dict
            Dictionary containing weights, bias, and losses
        """
        if self.weights is not None and self.losses is not None:
            return {
                'weights': self.weights,
                'bias': self.bias if self.add_bias else None,
                'losses': self.losses
            }
        else:
            raise ValueError("Model is not trained yet.")
        
    def get_weights(self):
        return self.weights, self.bias