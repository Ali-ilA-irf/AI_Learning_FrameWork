import numpy as np
import matplotlib.pyplot as plt
from Phase4_Step1 import X_train, y_train, X_test, y_test

class DeltaRule:
    def __init__(self, learning_rate=0.01, epochs=100):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0
        self.mse_history = []

    def fit(self, X, y):
        n_features = X.shape[1]
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.mse_history = []

        for epoch in range(self.epochs):
            # Forward pass (Batch)
            linear_output = np.dot(X, self.weights) + self.bias
            
            # Error
            errors = y - linear_output
            
            # Batch gradient descent weight update
            self.weights += self.lr * np.dot(X.T, errors) / len(X)
            self.bias    += self.lr * np.mean(errors)
            
            # Record MSE
            mse = np.mean(errors ** 2)
            self.mse_history.append(mse)

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{self.epochs} | MSE: {mse:.4f}")

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        # Threshold at 0.5 for classification since labels are 0 and 1
        return np.where(linear_output >= 0.5, 1, 0)

if __name__ == '__main__':
    print("\n--- Training Delta Rule (Batch Gradient Descent) ---\n")
    delta_rule = DeltaRule(learning_rate=0.05, epochs=100)
    delta_rule.fit(X_train, y_train)

    y_pred = delta_rule.predict(X_test)
    test_acc = np.mean(y_pred == y_test)
    print(f"\n--- Delta Rule Test Accuracy: {test_acc:.2%} ---\n")

    # Plot MSE
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, 101), delta_rule.mse_history, color='blue', linewidth=2)
    plt.title('Delta Rule Mean Squared Error over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('delta_rule_mse.png')
    print("\n  Plot saved as delta_rule_mse.png")
