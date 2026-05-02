import numpy as np
from Phase4_Step1 import X_train, y_train, y_array
# ---------------------------------------------
# HELPER - EUCLIDEAN DISTANCE
# ---------------------------------------------

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# ---------------------------------------------
# K-MEANS CLUSTERING FROM SCRATCH
# ---------------------------------------------

def kmeans(X, k, max_iters=100):

    np.random.seed(42)
    random_indices = np.random.choice(len(X), k, replace=False)
    centroids      = X[random_indices]

    for iteration in range(max_iters):

        labels   = []
        clusters = [[] for _ in range(k)]

        for i, point in enumerate(X):
            distances = [euclidean_distance(point, c) for c in centroids]
            nearest   = np.argmin(distances)
            clusters[nearest].append(i)
            labels.append(nearest)

        new_centroids = []
        for idx, cluster in enumerate(clusters):
            if cluster:
                new_centroids.append(X[cluster].mean(axis=0))
            else:
                new_centroids.append(centroids[idx])

        new_centroids = np.array(new_centroids)

        if np.allclose(centroids, new_centroids):
            print(f"  K-Means converged at iteration {iteration + 1}")
            break

        centroids = new_centroids

    return np.array(labels), centroids


k = len(np.unique(y_array))

if __name__ == '__main__':
    print(f"\n--- Running K-Means with k={k} ---\n")
km_labels, km_centroids = kmeans(X_train, k)

if __name__ == '__main__':
    print("\n---- K-Means Cluster Purity ----\n")
    for cluster_id in range(k):
        cluster_indices = np.where(km_labels == cluster_id)[0]
        true_labels     = y_train[cluster_indices]
        if len(true_labels) > 0:
            most_common = np.bincount(true_labels).argmax()
            purity      = np.sum(true_labels == most_common) / len(true_labels)
            print(f"  Cluster {cluster_id}: {len(cluster_indices)} points | Purity: {purity:.2%}")


# ---------------------------------------------
# K-MEDOID CLUSTERING FROM SCRATCH
# ---------------------------------------------

def kmedoids(X, k, max_iters=100):

    np.random.seed(42)
    medoid_indices = np.random.choice(len(X), k, replace=False)
    medoids        = X[medoid_indices]

    for iteration in range(max_iters):

        labels   = []
        clusters = [[] for _ in range(k)]

        for i, point in enumerate(X):
            distances = [euclidean_distance(point, m) for m in medoids]
            nearest   = np.argmin(distances)
            labels.append(nearest)
            clusters[nearest].append(i)

        new_medoids = []
        for idx, cluster in enumerate(clusters):
            if cluster:
                min_cost    = float('inf')
                best_medoid = medoids[idx]
                for ci in cluster:
                    cost = sum(euclidean_distance(X[ci], X[cj]) for cj in cluster)
                    if cost < min_cost:
                        min_cost    = cost
                        best_medoid = X[ci]
                new_medoids.append(best_medoid)
            else:
                new_medoids.append(medoids[idx])

        new_medoids = np.array(new_medoids)

        if np.allclose(medoids, new_medoids):
            print(f"  K-Medoid converged at iteration {iteration + 1}")
            break

        medoids = new_medoids

    return np.array(labels), medoids


# Running on 200 sample rows to speed it up
np.random.seed(42)
sample_idx  = np.random.choice(len(X_train), 200, replace=False)
X_sample    = X_train[sample_idx]
y_sample    = y_train[sample_idx]

if __name__ == '__main__':
    print(f"\n--- Running K-Medoid on 200 sample rows ---\n")
kmed_labels, kmed_medoids = kmedoids(X_sample, k)

if __name__ == '__main__':
    print("\n---- K-Medoid Cluster Purity ----\n")
    for cluster_id in range(k):
        cluster_indices = np.where(kmed_labels == cluster_id)[0]
        true_labels     = y_sample[cluster_indices]
        if len(true_labels) > 0:
            most_common = np.bincount(true_labels).argmax()
            purity      = np.sum(true_labels == most_common) / len(true_labels)
            print(f"  Cluster {cluster_id}: {len(cluster_indices)} points | Purity: {purity:.2%}")


# ---------------------------------------------
# WITHIN CLUSTER DISTANCE COMPARISON
# ---------------------------------------------

def within_cluster_distance(X, labels, centers):
    total = 0
    for i, point in enumerate(X):
        total += euclidean_distance(point, centers[labels[i]])
    return total

km_wcd  = within_cluster_distance(X_train,  km_labels,   km_centroids)
kmd_wcd = within_cluster_distance(X_sample, kmed_labels, kmed_medoids)

if __name__ == '__main__':
    print(f"\n---- Within Cluster Distance Comparison ----\n")
    print(f"  K-Means  WCD : {km_wcd:.2f}")
    print(f"  K-Medoid WCD : {kmd_wcd:.2f}")
    print(f"  {'K-Means' if km_wcd < kmd_wcd else 'K-Medoid'} produced tighter clusters")