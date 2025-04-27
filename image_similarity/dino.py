import torch
from transformers import ViTImageProcessor, ViTModel
from PIL import Image
import numpy as np

# Load the DINO model and image processor
processor = ViTImageProcessor.from_pretrained('facebook/dino-vitb16')
model = ViTModel.from_pretrained('facebook/dino-vitb16')
model.eval()

def extract_dino_features(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        features = outputs.last_hidden_state[:, 0, :].numpy()
    features = features.flatten()
    print(f"Feature norm before normalization: {np.linalg.norm(features)}")
    features = features / np.linalg.norm(features)
    print(f"Feature norm after normalization: {np.linalg.norm(features)}")
    return features