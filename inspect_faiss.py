import faiss
import numpy as np

# Load the FAISS index from disk
index = faiss.read_index("image_index.faiss")

# Print basic information about the index
print(f"Number of vectors in the index: {index.ntotal}")
print(f"Dimensionality of the vectors: {index.d}")

# Optionally, extract and print the vectors (if the index supports reconstruction)
try:
    # Reconstruct all vectors in the index
    vectors = np.zeros((index.ntotal, index.d), dtype=np.float32)
    for i in range(index.ntotal):
        vectors[i] = index.reconstruct(i)
    
    print("\nVectors stored in the index:")
    for i, vector in enumerate(vectors):
        print(f"Vector {i} (associated with FAISS internal ID {i}):")
        print(vector[:10], "... (first 10 dimensions)")  # Print first 10 dimensions for brevity
        print(f"Length of vector: {len(vector)}\n")
except AttributeError as e:
    print("This index type does not support vector reconstruction. Use a different method to inspect vectors.")