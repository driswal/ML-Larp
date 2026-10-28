from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

# Folder this script lives in, so it works no matter where you launch it from
HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()


def find_data(filename):
    """Find a data file next to this script or in any folder below it."""
    match = next(HERE.rglob(filename), None)
    if match is None:
        raise FileNotFoundError(f"Could not find '{filename}' in or below {HERE}")
    return match


# Load dataset
df = pd.read_csv(find_data("Mall_Customers.csv"))
print(df.head())

# Select and scale features
X = df[["Annual Income (k$)", "Spending Score (1-100)"]].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means clustering
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
y_means = kmeans.fit_predict(X_scaled)

# Agglomerative clustering
hc = AgglomerativeClustering(n_clusters=5)
y_hc = hc.fit_predict(X_scaled)

# Evaluation metrics
print("K-Means Inertia:", round(kmeans.inertia_, 2))
print("K-Means Silhouette Score:",
      round(silhouette_score(X_scaled, y_means), 4))
print("Agglomerative Silhouette Score:",
      round(silhouette_score(X_scaled, y_hc), 4))

# Convert centroids to original scale
centroids = scaler.inverse_transform(kmeans.cluster_centers_)

# Visualize clusters
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# K-Means plot
ax1.scatter(X[:, 0], X[:, 1], c=y_means, cmap="viridis", s=50)
ax1.scatter(centroids[:, 0], centroids[:, 1],
            c="red", marker="X", s=150, label="Centroids")
ax1.set_title("K-Means Clustering (K=5)")
ax1.set_xlabel("Annual Income (k$)")
ax1.set_ylabel("Spending Score (1-100)")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Agglomerative plot
ax2.scatter(X[:, 0], X[:, 1], c=y_hc, cmap="viridis", s=50)
ax2.set_title("Agglomerative Clustering (K=5)")
ax2.set_xlabel("Annual Income (k$)")
ax2.set_ylabel("Spending Score (1-100)")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Comparison
print("\nK-Means Strengths: Fast, efficient, and scalable.")
print("K-Means Weaknesses: Requires K in advance and is sensitive to outliers.")
print("Agglomerative Strengths: Reveals hierarchical relationships between clusters.")
print("Agglomerative Weaknesses: Computationally expensive and cannot undo merges.")
