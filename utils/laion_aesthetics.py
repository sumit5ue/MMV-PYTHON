import os
import torch
import torch.nn as nn
import clip
from PIL import Image
from urllib.request import urlretrieve
import json
import glob

# Function to download aesthetic model
def get_aesthetic_model(clip_model="vit_l_14"):
    home = os.path.expanduser("~")
    cache_folder = home + "/.cache/emb_reader"
    path_to_model = cache_folder + "/sa_0_4_" + clip_model + "_linear.pth"
    if not os.path.exists(path_to_model):
        os.makedirs(cache_folder, exist_ok=True)
        url_model = (
            "https://github.com/LAION-AI/aesthetic-predictor/raw/main/sa_0_4_"
            + clip_model
            + "_linear.pth"
        )
        print(f"Downloading aesthetic model to {path_to_model}...")
        urlretrieve(url_model, path_to_model)
    m = nn.Linear(768, 1) if clip_model == "vit_l_14" else nn.Linear(512, 1)
    s = torch.load(path_to_model, map_location="cpu")
    m.load_state_dict(s)
    m.eval()
    return m

# Set device (try MPS, fallback to CPU)
device = "mps" if torch.backends.mps.is_available() else "cpu"
if device == "mps":
    try:
        # Test MPS with a simple operation
        test_tensor = torch.tensor([1.0], dtype=torch.float32, device=device)
        test_tensor + test_tensor
        print("MPS device is available and functional.")
    except Exception as e:
        print(f"MPS device failed: {e}. Falling back to CPU.")
        device = "cpu"
else:
    print("MPS not available. Using CPU.")

# Load CLIP model
try:
    clip_model, preprocess = clip.load("ViT-L/14", device=device)
    clip_model = clip_model.float()  # Ensure float32 precision
except Exception as e:
    print(f"Failed to load CLIP model: {e}")
    exit(1)

# Load aesthetic model
try:
    aesthetic_model = get_aesthetic_model(clip_model="vit_l_14").to(device)
    aesthetic_model = aesthetic_model.float()  # Ensure float32 precision
except Exception as e:
    print(f"Failed to load aesthetic model: {e}")
    exit(1)

# Preprocess and embed
def get_aesthetic_score(image_path):
    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = clip_model.encode_image(image).float()  # Ensure float32
        score = aesthetic_model(embedding).cpu().numpy().item()  # Raw scalar score
    return score

# Main processing
if __name__ == "__main__":
    # Set Downloads folder path
    downloads_folder = os.path.expanduser("~/Downloads/abundant")
    output_json = os.path.join(downloads_folder, "aesthetic_scores.json")

    # Find all .jpg and .png files
    image_extensions = ["*.jpg", "*.jpeg", "*.png"]
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(downloads_folder, ext)))

    if not image_paths:
        print(f"No images found in {downloads_folder}")
        exit(0)

    # Process images and collect scores
    scores = {}
    for image_path in image_paths:
        filename = os.path.basename(image_path)
        try:
            score = get_aesthetic_score(image_path)
            scores[filename] = float(score)  # Ensure JSON compatibility
            print(f"Aesthetic Score for {filename}: {score:.2f}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    # Sort scores by value (ascending) and convert to list of dicts
    sorted_scores = [
        {"filename": filename, "score": score}
        for filename, score in sorted(scores.items(), key=lambda x: x[1])
    ]

    # Save sorted scores to JSON
    try:
        with open(output_json, "w") as f:
            json.dump(sorted_scores, f, indent=4)
        print(f"Saved sorted scores to {output_json}")
    except Exception as e:
        print(f"Error saving JSON: {e}")