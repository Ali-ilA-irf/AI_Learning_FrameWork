import numpy as np
import matplotlib.pyplot as plt
from Phase4_Step1 import X_train, y_train, X_test, y_test

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def sigmoid(x):
    x_clipped = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x_clipped))

class MLP:
    def __init__(self, input_size, hidden_size1=16, hidden_size2=8, learning_rate=0.1, epochs=200):
        self.lr = learning_rate
        self.epochs = epochs
        self.loss_history = []
        self.accuracy_history = []
        
        # Initialize weights
        np.random.seed(42)
        self.W1 = np.random.randn(input_size, hidden_size1) * 0.1
        self.b1 = np.zeros((1, hidden_size1))
        
        self.W2 = np.random.randn(hidden_size1, hidden_size2) * 0.1
        self.b2 = np.zeros((1, hidden_size2))
        
        self.W3 = np.random.randn(hidden_size2, 1) * 0.1
        self.b3 = np.zeros((1, 1))

    def fit(self, X, y):
        y = y.reshape(-1, 1)
        m = X.shape[0]
        
        for epoch in range(self.epochs):
            # --- Forward Propagation ---
            Z1 = np.dot(X, self.W1) + self.b1
            A1 = relu(Z1)
            
            Z2 = np.dot(A1, self.W2) + self.b2
            A2 = relu(Z2)
            
            Z3 = np.dot(A2, self.W3) + self.b3
            A3 = sigmoid(Z3)
            
            # --- Loss Calculation (Binary Cross Entropy) ---
            epsilon = 1e-15
            loss = -np.mean(y * np.log(A3 + epsilon) + (1 - y) * np.log(1 - A3 + epsilon))
            self.loss_history.append(loss)
            
            # Accuracy
            predictions = (A3 >= 0.5).astype(int)
            accuracy = np.mean(predictions == y)
            self.accuracy_history.append(accuracy)
            
            # --- Backpropagation ---
            # Output layer
            dZ3 = A3 - y
            dW3 = np.dot(A2.T, dZ3) / m
            db3 = np.sum(dZ3, axis=0, keepdims=True) / m
            
            # Hidden layer 2
            dA2 = np.dot(dZ3, self.W3.T)
            dZ2 = dA2 * relu_derivative(Z2)
            dW2 = np.dot(A1.T, dZ2) / m
            db2 = np.sum(dZ2, axis=0, keepdims=True) / m
            
            # Hidden layer 1
            dA1 = np.dot(dZ2, self.W2.T)
            dZ1 = dA1 * relu_derivative(Z1)
            dW1 = np.dot(X.T, dZ1) / m
            db1 = np.sum(dZ1, axis=0, keepdims=True) / m
            
            # --- Update Weights ---
            self.W3 -= self.lr * dW3
            self.b3 -= self.lr * db3
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2
            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{self.epochs} | Loss: {loss:.4f} | Accuracy: {accuracy:.2%}")

    def predict(self, X):
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = relu(Z1)
        Z2 = np.dot(A1, self.W2) + self.b2
        A2 = relu(Z2)
        Z3 = np.dot(A2, self.W3) + self.b3
        A3 = sigmoid(Z3)
        return (A3 >= 0.5).astype(int).flatten()

if __name__ == '__main__':
    print("\n--- Training Multilayer Perceptron (MLP) ---\n")
    mlp = MLP(input_size=X_train.shape[1], learning_rate=0.1, epochs=200)
    mlp.fit(X_train, y_train)

    y_pred = mlp.predict(X_test)
    test_acc = np.mean(y_pred == y_test)
    print(f"\n--- MLP Test Accuracy: {test_acc:.2%} ---\n")

    # Plot Loss
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, 201), mlp.loss_history, color='green', linewidth=2)
    plt.title('MLP Cross-Entropy Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('mlp_loss.png')
    print("\n  Plot saved as mlp_loss.png")
