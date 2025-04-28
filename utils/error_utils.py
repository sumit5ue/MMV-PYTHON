# utils/error_utils.py

import json
import os
import traceback
from config import get_errors_path

def log_error(photoId, faceId, step, errorMessage, traceback, modelType, path, partner="unknown"):
    error_path = get_errors_path(partner)
    os.makedirs(os.path.dirname(error_path), exist_ok=True)

    entry = {
        "photoId": photoId,
        "faceId": faceId,
        "step": step,
        "errorMessage": errorMessage,
        "traceback": traceback,
        "modelType": modelType,
        "path": path
    }
    with open(error_path, "a") as f:
        f.write(json.dumps(entry) + "\n")
