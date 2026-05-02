import numpy as np
import matplotlib.pyplot as plt

# Import data
from Phase4_Step1 import X_train, y_train, X_test, y_test, y_array, df
# Import clustering functions and results
from Phase4_Step2 import kmeans, kmedoids, within_cluster_distance, k, km_labels, km_centroids, kmed_labels, kmed_medoids, km_wcd, kmd_wcd, X_sample, y_sample
# Import models
from Phase4_Step3 import Perceptron
from Phase4_Step4 import DeltaRule
from Phase4_Step5 import MLP

print("=" * 70)
print("PHASE 4: MACHINE LEARNING RESULTS")
print("=" * 70)

# =============================================================================
# STEP 1: PREPARE DATA
# =============================================================================
print("\n[STEP 1: PREPARE DATA FOR MACHINE LEARNING]")
print("  Missing values handled, categorical variables encoded, numerical normalized.")
print(f"  Train shape: X={X_train.shape}, y={y_train.shape}")
print(f"  Test shape : X={X_test.shape}, y={y_test.shape}")

# =============================================================================
# STEP 2: UNSUPERVISED LEARNING
# =============================================================================
print("\n[STEP 2: UNSUPERVISED LEARNING]")

# K-Means
print(f"\n--- K-Means Clustering (k={k}) ---")
for cluster_id in range(k):
    cluster_indices = np.where(km_labels == cluster_id)[0]
    true_labels = y_train[cluster_indices]
    if len(true_labels) > 0:
        most_common = np.bincount(true_labels).argmax()
        purity = np.sum(true_labels == most_common) / len(true_labels)
        print(f"  Cluster {cluster_id}: {len(cluster_indices):>4} points | Purity: {purity:.2%}")

# K-Medoid
print(f"\n--- K-Medoid Clustering (k={k}, sample size 200) ---")
for cluster_id in range(k):
    cluster_indices = np.where(kmed_labels == cluster_id)[0]
    true_labels = y_sample[cluster_indices]
    if len(true_labels) > 0:
        most_common = np.bincount(true_labels).argmax()
        purity = np.sum(true_labels == most_common) / len(true_labels)
        print(f"  Cluster {cluster_id}: {len(cluster_indices):>4} points | Purity: {purity:.2%}")

# WCD Comparison
print(f"\n--- Within-Cluster Distance (WCD) ---")
print(f"  K-Means WCD  : {km_wcd:.2f}")
print(f"  K-Medoid WCD : {kmd_wcd:.2f}")

# =============================================================================
# STEP 3: PERCEPTRON
# =============================================================================
print("\n[STEP 3: PERCEPTRON TRAINING]")
perceptron = Perceptron(learning_rate=0.01, epochs=50)
# Silence internal prints
import sys, os
old_stdout = sys.stdout
sys.stdout = open(os.devnull, 'w')
perceptron.fit(X_train, y_train)
sys.stdout = old_stdout

y_pred_p = perceptron.predict(X_test)
acc_p = np.mean(y_pred_p == y_test)
print(f"  Test Accuracy: {acc_p:.2%}")

# =============================================================================
# STEP 4: DELTA RULE (GRADIENT DESCENT)
# =============================================================================
print("\n[STEP 4: DELTA RULE / GRADIENT DESCENT]")
delta = DeltaRule(learning_rate=0.05, epochs=100)
old_stdout = sys.stdout
sys.stdout = open(os.devnull, 'w')
delta.fit(X_train, y_train)
sys.stdout = old_stdout

y_pred_d = delta.predict(X_test)
acc_d = np.mean(y_pred_d == y_test)
print(f"  Test Accuracy: {acc_d:.2%}")
print(f"  Final MSE: {delta.mse_history[-1]:.4f}")

# =============================================================================
# STEP 5: MULTILAYER PERCEPTRON (BACKPROPAGATION)
# =============================================================================
print("\n[STEP 5: MULTILAYER PERCEPTRON (MLP)]")
mlp = MLP(input_size=X_train.shape[1], learning_rate=0.1, epochs=200)
old_stdout = sys.stdout
sys.stdout = open(os.devnull, 'w')
mlp.fit(X_train, y_train)
sys.stdout = old_stdout

y_pred_mlp = mlp.predict(X_test)
acc_mlp = np.mean(y_pred_mlp == y_test)
print(f"  Test Accuracy: {acc_mlp:.2%}")
print(f"  Final Loss: {mlp.loss_history[-1]:.4f}")

# =============================================================================
# STEP 6: COMPARE ALL MODELS
# =============================================================================
print("\n[STEP 6: FINAL COMPARISON TABLE]")
print("-" * 75)
print(f"{'Model':<20} {'Accuracy':<15} {'Notes'}")
print("-" * 75)
print(f"{'K-Means':<20} {'Unsupervised':<15} {'WCD: ' + str(round(km_wcd, 2))}")
print(f"{'K-Medoid':<20} {'Unsupervised':<15} {'WCD: ' + str(round(kmd_wcd, 2))}")
print(f"{'Perceptron':<20} {acc_p:<15.2%} {'Single neuron, step act.'}")
print(f"{'Delta Rule':<20} {acc_d:<15.2%} {'Linear act., MSE minimizing'}")
print(f"{'MLP (Backprop)':<20} {acc_mlp:<15.2%} {'2 Hidden layers, ReLU, Sigmoid'}")
print("-" * 75)
print("=" * 70)
