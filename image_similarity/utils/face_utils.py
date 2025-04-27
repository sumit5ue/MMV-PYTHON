import cv2
import numpy as np
from insightface.app import FaceAnalysis

app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=0)

def detect_faces(image_path):
    img = cv2.imread(image_path)
    faces_raw = app.get(img)
    faces = []
    for face in faces_raw:
        faces.append({
            "bbox": face.bbox.astype(int).tolist(),
            "quality": float(face.det_score),
            "embedding": face.embedding
        })
    return faces

def filter_faces_by_quality(faces, min_quality=70, min_size=40):
    filtered = []
    for face in faces:
        x1, y1, x2, y2 = face["bbox"]
        w = x2 - x1
        h = y2 - y1
        if face["quality"] >= min_quality and w >= min_size and h >= min_size:
            filtered.append(face)
    return filtered


def extract_face_embedding(face):
    return np.array(face["embedding"]).astype("float32").reshape(1, -1)
