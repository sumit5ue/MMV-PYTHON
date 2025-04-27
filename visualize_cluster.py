import numpy as np
import umap
import matplotlib.pyplot as plt
import json
import time
import os

# Start timing
start_time = time.time()

# Load embeddings
embeddings = np.load("/Users/sumit/mmv-python/clip_embeddings/vectors.npy")
print(f"Embeddings shape: {embeddings.shape}")

# Load cluster labels
with open("/Users/sumit/mmv-python/clip_embeddings/vectors_clusters.json") as f:
    clusters_json = json.load(f)

# Create a filepath-to-index mapping
# Replace this with your actual mapping (e.g., from the embedding generation script)
# Example: {'/path/to/image_123.jpg': 123, '/path/to/1LQS_1LQS14_<UUID>.jpg': 0, ...}
filepath_to_index = {}
# Placeholder: You need to provide the correct mapping
# For demonstration, we'll extract indices from 'image_<number>.jpg' and warn about UUIDs
for i, filepath in enumerate(clusters_json["clusters"]["-1"]):  # Use a cluster for testing
    filename = os.path.basename(filepath["filepath"])
    if filename.startswith("image_") and filename.endswith(".jpg"):
        try:
            idx = int(filename[len("image_"):-len(".jpg")])
            filepath_to_index[filepath["filepath"]] = idx
        except ValueError:
            print(f"Warning: Cannot extract index from {filename}")
    else:
        print(f"Warning: Non-numeric filename {filename}, needs manual index assignment")

# Map vector index to cluster
index_to_cluster = {}
for cluster_id, items in clusters_json["clusters"].items():
    for item in items:
        filepath = item["filepath"]
        idx = filepath_to_index.get(filepath)
        if idx is None:
            print(f"Error: No index for {filepath}. Please provide a filepath-to-index mapping.")
            continue
        index_to_cluster[idx] = int(cluster_id)

# Assign cluster labels
labels = [index_to_cluster.get(i, -1) for i in range(len(embeddings))]

# Reduce to 2D
reducer = umap.UMAP(n_components=2, random_state=42)
embedding_2d = reducer.fit_transform(embeddings)

# Plot
plt.figure(figsize=(12, 8))
scatter = plt.scatter(
    embedding_2d[:, 0],
    embedding_2d[:, 1],
    c=labels,
    cmap="tab20",
    s=10
)
plt.colorbar(scatter, label="Cluster ID")
plt.title("2D Visualization of Clusters")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.grid(True)
plt.show()

# Print runtime
print(f"Total runtime: {time.time() - start_time:.2f} seconds")