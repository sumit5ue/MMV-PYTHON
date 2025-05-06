from PIL import Image
import pillow_heif

# Explicitly register pillow-heif plugin
pillow_heif.register_heif_opener()

import os

# Use full path based on current script location
script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, "test.HEIC")

try:
    img = Image.open(image_path)
    img.show()
except Exception as e:
    print(f"Error: {e}")
