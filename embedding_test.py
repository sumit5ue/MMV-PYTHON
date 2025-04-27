import cv2
import numpy as np
from insightface.app import FaceAnalysis

face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=0)

img1 = cv2.imread("/Users/sumit/Downloads/photo1.jpg")
img2 = cv2.imread("/Users/sumit/Downloads/f2caf8e9-6101-4094-91cc-07e1f593986a.jpg")
               
faces1 = face_app.get(img1)
faces2 = face_app.get(img2)

if not faces1 or not faces2:
    print("No faces detected")
    exit()
emb1 = faces1[0].normed_embedding# / np.linalg.norm(faces1[0].embedding)
emb2 = faces2[0].normed_embedding #/ np.linalg.norm(faces2[0].embedding)
similarity = np.dot(emb1, emb2)
# print("emb1",emb1)
# print("emb2",emb2)
print(f"Manual cosine similarity: {similarity}")
print(f"Face 1: det_score={faces1[0].det_score}, pose={faces1[0].pose}")
print(f"Face 2: det_score={faces2[0].det_score}, pose={faces2[0].pose}")