Assignment3

## packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering
import umap

## dataset
cancer = load_breast_cancer()

X = cancer.data
y = cancer.target
feature_names = cancer.feature_names
target_names = cancer.target_names

df = pd.DataFrame(X, columns=feature_names)
df["diagnosis"] = y
df["diagnosis_label"] = df["diagnosis"].map({0: "malignant", 1: "benign"})

df.head()

## scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

## dimensionality reduction
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_scaled)

umap_model = umap.UMAP(n_components=2, random_state=42)
X_umap = umap_model.fit_transform(X_scaled)

## clustering
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)

hierarchical = AgglomerativeClustering(n_clusters=2)
hierarchical_labels = hierarchical.fit_predict(X_scaled)

cluster_labels = kmeans_labels

## figure 1

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

projections = [
    (X_pca, "PCA"),
    (X_tsne, "t-SNE"),
    (X_umap, "UMAP")
]

for ax, (projection, title) in zip(axes, projections):
    scatter = ax.scatter(
        projection[:, 0],
        projection[:, 1],
        c=y,
        alpha=0.7
    )
    ax.set_title(f"{title} Projection Colored by True Diagnosis")
    ax.set_xlabel(f"{title} 1")
    ax.set_ylabel(f"{title} 2")

handles, labels = scatter.legend_elements()
fig.legend(handles, ["Malignant", "Benign"], title="Diagnosis", loc="upper right")

plt.tight_layout()
plt.savefig("Assignment3_Figure1.png", dpi=300, bbox_inches="tight")
plt.show()

## figure 2

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (projection, title) in zip(axes, projections):
    scatter = ax.scatter(
        projection[:, 0],
        projection[:, 1],
        c=cluster_labels,
        alpha=0.7
    )
    ax.set_title(f"{title} Projection Colored by K-Means Cluster")
    ax.set_xlabel(f"{title} 1")
    ax.set_ylabel(f"{title} 2")

handles, labels = scatter.legend_elements()
fig.legend(handles, ["Cluster 0", "Cluster 1"], title="K-Means Cluster", loc="upper right")

plt.tight_layout()
plt.savefig("Assignment3_Figure2.png", dpi=300, bbox_inches="tight")
plt.show()

