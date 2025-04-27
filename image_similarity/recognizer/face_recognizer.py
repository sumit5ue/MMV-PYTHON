import os
import numpy as np
import logging
from image_similarity.utils.face_utils import detect_faces
from numpy.linalg import norm
from numpy import dot

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return float(dot(a, b) / (norm(a) * norm(b)))  # Cast to float early

def index_known_faces(folder="image_similarity/recognizer/known_faces"):
    """
    Loads all known faces from the given folder and returns a dictionary:
    {
        "Alice": [embedding1, embedding2, ...]
    }
    """
    known_faces = {}

    logger.info(f"🔍 Indexing faces from: {folder}")

    for filename in os.listdir(folder):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        path = os.path.join(folder, filename)
        faces = detect_faces(path)
        if not faces:
            logger.warning(f"❌ No face found in {filename}")
            continue

        name = os.path.splitext(filename)[0].capitalize()
        emb = faces[0]["embedding"]

        if name not in known_faces:
            known_faces[name] = []
        known_faces[name].append(emb)

        logger.info(f"✅ Indexed: {name} (1 face from {filename})")

    if not known_faces:
        logger.warning("⚠️ No known faces were indexed.")
    else:
        logger.info(f"✅ Total known individuals: {len(known_faces)}")

    return known_faces

def recognize_faces_in_image(image_path, known_faces, threshold=0.6):
    """
    Detects all faces in an image and compares them to the known faces.
    Returns a list of face detections with similarity scores.
    """
    detected = detect_faces(image_path)
    results = []

    logger.info(f"\n🔍 Detected {len(detected)} face(s) in {os.path.basename(image_path)}")

    for i, face in enumerate(detected):
        logger.debug(f"→ Face {i+1} bbox: {face['bbox']}")
        scores = []
        for name, embeddings in known_faces.items():
            sim_scores = [cosine_similarity(face["embedding"], emb) for emb in embeddings]
            best_sim = max(sim_scores)
            scores.append({
                "name": name,
                "similarity": round(best_sim, 4),
                "match": bool(best_sim >= threshold)
            })
        results.append({
            "bbox": face["bbox"],
            "matches": sorted(scores, key=lambda x: -x["similarity"])
        })

    return results
