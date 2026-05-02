import numpy as np
import matplotlib.pyplot as plt
from Phase4_Step1 import X_train, y_train, y_array, X_test, y_test
from Phase4_Step2 import kmd_wcd

class Perceptron:
    def __init__(self, learning_rate=0.01, epochs=50):
        self.lr              = learning_rate
        self.epochs          = epochs
        self.weights         = None
        self.bias            = 0
        self.accuracy_history = []

    def step_function(self, x):
        return 1 if x > 0 else 0

    def fit(self, X, y):
        n_features            = X.shape[1]
        self.weights          = np.zeros(n_features)
        self.bias             = 0
        self.accuracy_history = []

        for epoch in range(self.epochs):
            correct = 0

            for i in range(len(X)):

                # Forward pass
                linear_output = np.dot(X[i], self.weights) + self.bias
                prediction    = self.step_function(linear_output)

                # Compute error
                error         = y[i] - prediction

                # Update weights and bias
                self.weights += self.lr * error * X[i]
                self.bias    += self.lr * error

                if prediction == y[i]:
                    correct += 1

            accuracy = correct / len(X)
            self.accuracy_history.append(accuracy)

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{self.epochs} | Accuracy: {accuracy:.2%}")

    def predict(self, X):
        results = []
        for i in range(len(X)):
            linear_output = np.dot(X[i], self.weights) + self.bias
            results.append(self.step_function(linear_output))
        return np.array(results)


if __name__ == '__main__':
    # ---------------------------------------------
    # TRAIN PERCEPTRON
    # ---------------------------------------------
    print("\n--- Training Perceptron ---\n")
    perceptron = Perceptron(learning_rate=0.01, epochs=50)
    perceptron.fit(X_train, y_train)


    # ---------------------------------------------
    # TEST ACCURACY
    # ---------------------------------------------
    y_pred   = perceptron.predict(X_test)
    test_acc = np.mean(y_pred == y_test)

    print(f"\n--- Perceptron Test Accuracy: {test_acc:.2%} ---\n")


    # ---------------------------------------------
    # PLOT TRAINING ACCURACY OVER EPOCHS
    # ---------------------------------------------
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, 51),
             perceptron.accuracy_history,
             color='purple',
             linewidth=2,
             marker='o',
             markersize=3)
    plt.title('Perceptron Training Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('perceptron_accuracy.png')
    plt.show()
    print("\n  Plot saved as perceptron_accuracy.png")

    # ---------------------------------------------
    # FINAL COMPARISON TABLE
    # ---------------------------------------------
    print("\n")
    print("-" * 70)
    print(f"{'Model':<28} {'Accuracy':<20} {'Notes'}")
    print("-" * 70)
    print(f"{'Perceptron':<28} {test_acc:<20.2%} {'Linear classifier from scratch'}")
    print(f"{'K-Means':<28} {'N/A (purity above)':<20} {'Unsupervised clustering'}")
    print(f"{'K-Medoid':<28} {'N/A (purity above)':<20} {'WCD: ' + str(round(kmd_wcd, 2))}")
    print("-" * 70)
