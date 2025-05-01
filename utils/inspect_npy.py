import numpy as np

path = "/Users/sumit/Documents/ai_analysis/67c5078cfb7ebb148255e19e/embeddings/67c5078cfb7ebb148255e19e_clip.npy"

arr = np.load(path)

print(f"Shape: {arr.shape}")
print(f"Dtype: {arr.dtype}")
print(f"Number of embeddings = {arr.shape[0]}")
print(f"First embedding sample: {arr[0][:5]}")  # print first 5 dimensions of first vector
