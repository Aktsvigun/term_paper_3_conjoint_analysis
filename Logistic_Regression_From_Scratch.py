import numpy as np

class LogisticRegression:
    def __init__(self, intercept=True, l1=0, l2=0, weights=None, threshold=0.5):
        self.intercept = intercept
        self.l1 = l1
        self.l2 = l2
        self.weights = weights
        self.threshold = threshold
    
    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    @staticmethod
    def __add_intercept(X):
        return np.hstack((np.ones((len(X), 1)), X))
    
    def __loss(self, w, X, y):
        return -(y.T @ np.log(self.__sigmoid(X@w)) +
                 (1 - y).T @ np.log(self.__sigmoid(-X@w)))
    
    def __gradient_loss(self, w, X, y):
        return (X.T@(self.__sigmoid(X@w) - y[:, None]))
    
    def fit(self, X, y, learning_rate=0.01, num_epoch=10000, verbose=False):
        if self.intercept:
            X = LogisticRegression.__add_intercept(X)
        # Initializing weights
        self.weights = np.zeros((X.shape[1], 1)) if isinstance(self.weights, type(None)) else self.weights
        
        for epoch in range(1, num_epoch + 1):
            self.weights -= learning_rate * self.__gradient_loss(self.weights, X, y)
            if verbose and epoch % 1e+4 == 0:
                print(f'Epoch: {epoch}; loss: {self.__loss(self.weights, X, y)}; weights: {self.weights}.')
                
    def predict_proba(self, X):
        if self.intercept:
            X = LogisticRegression.__add_intercept(X)
        return self.__sigmoid(X@self.weights)
    
    def predict(self, X):
        return self.predict_proba(X) >= self.threshold
    
    def get_weights(self):
        return np.ravel(self.weights)