import numpy as np
import random

author = "t.me/Bengamin_Button t.me/XillenAdapter"
print(author)

class SimpleML:
    def __init__(self):
        self.weights = None
        self.bias = 0
        
    def train(self, X, y, epochs=1000, lr=0.01):
        n_samples, n_features = X.shape
        self.weights = np.random.randn(n_features)
        for epoch in range(epochs):
            y_pred = self.predict(X)
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            self.weights -= lr * dw
            self.bias -= lr * db
            if epoch % 100 == 0:
                loss = np.mean((y_pred - y) ** 2)
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
y = np.array([3, 5, 7, 9])
model = SimpleML()
print("Обучение модели...")
model.train(X, y)
print("Предсказание для [5, 6]:", model.predict(np.array([[5, 6]]))[0])

