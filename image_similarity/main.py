# face_pipeline_api.py

from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import uvicorn
import os
import cv2
import numpy as np
import traceback
from insightface.app import FaceAnalysis
import faiss
import json
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize
import uuid
from fastapi import APIRouter, UploadFile, File, Form
import faiss
import time
import joblib
import hdbscan
import torch
import clip
from PIL import Image

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ← you can replace * with ["http://localhost:3000"] for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/images", StaticFiles(directory="/Users/sumit/Documents/ranking_images2"), name="images")

# Initialize InsightFace
face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=0)
print("Models loaded:", face_app.models.keys())

device = "mps" if torch.backends.mps.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)

# Base directory for crops and metadata
BASE_FACE_DIR = "faces"
FAISS_INDEX_DIR = "faiss_indices"
META_FILE = "faiss_data/meta.jsonl"

# Utility for saving JSON

def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

# Append line to JSONL

def append_jsonl(data, path):
    with open(path, "a") as f:
        f.write(json.dumps(data) + "\n")

# Error wrapper

def structured_error(stage, photo_id, message, trace):
    return {
        "error": True,
        "stage": stage,
        "photoId": photo_id,
        "message": message,
        "trace": trace
    }



class DetectFolderRequest(BaseModel):
    folderPath: str
    collection: str

@app.post("/detect-faces-folder")
async def detect_faces_folder(req: DetectFolderRequest):
    try:
        print(f"🧪 detect_faces_folder called with: {req.folderPath}, {req.collection}")

        if not os.path.exists(req.folderPath):
            return {"error": f"Folder {req.folderPath} does not exist"}

        # Get all image files in the folder (extract filenames)
        image_files = [
            os.path.join(req.folderPath, f)
            for f in os.listdir(req.folderPath)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]

        print(f"Image Files: {image_files}")  # Debug log for all images in the folder

        if not image_files:
            return {"error": "No images found in folder"}

        # Path to your existing JSONL file
        jsonl_file_path = os.path.join("embeddings", f"{req.collection}.jsonl")

        # Load the existing JSONL data into a set of image filenames for quick lookup
        processed_images = set()
        if os.path.exists(jsonl_file_path):
            with open(jsonl_file_path, 'r') as file:
                for line in file:
                    data = json.loads(line)
                    # Extract only the filename from sourceImagePath for comparison
                    source_image_filename = os.path.basename(data.get("sourceImagePath", ""))
                    processed_images.add(source_image_filename.lower())  # Case-insensitive comparison

        # Process only the images that are not already in the JSONL file
        unprocessed_images = [
            image for image in image_files if os.path.basename(image).lower() not in processed_images
        ]

        print(f"Unprocessed Images: {unprocessed_images}")  # Debug log for unprocessed images

        if not unprocessed_images:
            return {"status": "no new images to process"}

        all_results = []
        for image_path in unprocessed_images:
            detect_req = DetectRequest(imagePath=image_path, collection=req.collection)
            result = await detect_faces(detect_req)
            # Save or log the result if needed
            all_results.append({
                "image": image_path,
                "result": result
            })

        return {"status": "completed", "filesProcessed": len(unprocessed_images)}

    except Exception as e:
        return structured_error("detect_faces_folder", "batch", str(e), traceback.format_exc())
      
class DetectRequest(BaseModel):
    imagePath: str
    collection: str

@app.post("/detect-faces")
async def detect_faces(req: DetectRequest):
    try:
        print(f"🧪 detect_faces called with: {req.imagePath}, {req.collection}")

        image = cv2.imread(req.imagePath)
        if image is None:
            return {"error": "Failed to load image", "path": req.imagePath}

        faces = face_app.get(image)
        print(f"[detect_faces] Found {len(faces)} faces in {req.imagePath}")

        result = []
        new_embeddings = []
        photo_id = str(uuid.uuid4())

        output_folder = os.path.join(BASE_FACE_DIR, req.collection)
        os.makedirs(output_folder, exist_ok=True)

       
        for i, face in enumerate(faces):
            # Crop using bbox
            x1, y1, x2, y2 = face.bbox.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)

            crop_img = image[y1:y2, x1:x2]
            if crop_img.size == 0:
                print(f"❌ Skipping face {i} — empty bbox crop")
                continue

            crop_path = os.path.join(output_folder, f"{photo_id}_{i}.jpg")
            cv2.imwrite(crop_path, crop_img)
            print(f"✅ Saved crop {i} to {crop_path}")

            # Validate and normalize embedding
            embedding = face.normed_embedding
            if embedding is None or len(embedding) == 0:
                print(f"❌ Skipping face {i} — missing embedding")
                continue
            
            embedding = embedding / np.linalg.norm(embedding)

            norm = np.linalg.norm(embedding)
            print(f"Face {i}: Embedding norm = {norm}")  # This should be very close to 1 if normalized
            print(f"Embedding shape: {embedding.shape}")
            # norm = np.linalg.norm(embedding)
            # print(f"Face {i}: Embedding norm = {norm}, Shape = {embedding.shape}")
            # embedding = embedding / norm if norm > 0 else embedding
            # print(f"Face {i}: Normalized embedding norm = {np.linalg.norm(embedding)}")

            result.append({
                "photoId": photo_id,
                "faceIndex": i,
                "bbox": face.bbox.tolist(),
                "poseType": classify_pose(*face.pose),
                "detScore": float(face.det_score),
                "cropPath": crop_path,
                "sourceImagePath": req.imagePath   # 🆕 ADD THIS
            })

            new_embeddings.append(embedding)

        if not new_embeddings:
            return {"warning": "Faces detected, but all crops or embeddings failed."}

        # Save embeddings
        npy_path = os.path.join("embeddings", f"{req.collection}.npy")
        os.makedirs(os.path.dirname(npy_path), exist_ok=True)

        new_embeddings = np.array(new_embeddings, dtype="float32")
        if os.path.exists(npy_path):
            existing = np.load(npy_path)
            existing = existing / np.linalg.norm(existing, axis=1, keepdims=True)

            # Normalize existing embeddings if needed
            # norms = np.linalg.norm(existing, axis=1)
            # if not np.allclose(norms, 1.0, rtol=0.01):
            #     existing = existing / np.linalg.norm(existing, axis=1, keepdims=True)
            all_embeddings = np.concatenate([existing, new_embeddings], axis=0)
        else:
            all_embeddings = new_embeddings

        np.save(npy_path, all_embeddings)
        print(f"💾 Saved {len(new_embeddings)} embeddings → {npy_path}")

        # Create FAISS index
        index = faiss.IndexFlatIP(all_embeddings.shape[1])
        index.add(all_embeddings)  # No faiss.normalize_L2
        index_path = os.path.join("embeddings", f"{req.collection}.index")
        faiss.write_index(index, index_path)
        print(f"📦 FAISS index saved → {index_path}")

        # Save metadata
        jsonl_path = os.path.join("embeddings", f"{req.collection}.jsonl")
        existing_count = 0
        if os.path.exists(jsonl_path):
            with open(jsonl_path) as f:
                existing_count = sum(1 for _ in f)

        with open(jsonl_path, "a") as f:
            for i, r in enumerate(result):
                r["vectorId"] = existing_count + i
                f.write(json.dumps(r) + "\n")

        print(f"📋 Appended {len(result)} entries to {jsonl_path}")
        return result

    except Exception as e:
        return structured_error("detect_faces", "unknown", str(e), traceback.format_exc())


@app.post("/find-similar")
async def find_similar_faces(
    file: UploadFile = File(...),
    collection: str = Form(...),
    top_k: int = Form(25)
):
    from numpy.linalg import norm

    # Read image
    image_bytes = await file.read()
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    faces = face_app.get(img)
    if not faces:
        return {"error": "No face detected"}

    query_embedding = faces[0].normed_embedding
    query_embedding = query_embedding / np.linalg.norm(query_embedding)
    norm = np.linalg.norm(query_embedding)
    print(f"Query embedding norm: {norm}, Shape: {query_embedding.shape}")
    # query_embedding = query_embedding / norm if norm > 0 else query_embedding
    # print(f"Query normalized embedding norm: {np.linalg.norm(query_embedding)}")
    
    # Paths
    base_path = os.path.join("embeddings", collection)
    index_path = f"{base_path}.index"
    jsonl_path = f"{base_path}.jsonl"

    if not os.path.exists(index_path):
        return {"error": f"FAISS index not found for collection '{collection}'"}

    if not os.path.exists(jsonl_path):
        return {"error": f"Metadata not found for collection '{collection}'"}

    # Load FAISS index
    faiss_index = faiss.read_index(index_path)

    # Search
    D, I = faiss_index.search(np.array([query_embedding], dtype="float32"), k=top_k)

    # Load metadata
    with open(jsonl_path) as f:
        metadata = [json.loads(line) for line in f]

    # Prepare result
    matches = []
    for score, idx in zip(D[0], I[0]):
        if idx >= len(metadata): continue
        matches.append({
            "vectorId": metadata[idx]["vectorId"],
            "photoId": metadata[idx]["photoId"],
            "cropPath": metadata[idx]["cropPath"],
            "sourceImagePath": metadata[idx]["sourceImagePath"],
            "similarity": round(float(score), 4),
            "detScore":float(metadata[idx]["detScore"])
                            

        })

    return {"matches": matches}

# Pose classification logic

def classify_pose(yaw, pitch, roll):
    if abs(yaw) < 15 and abs(pitch) < 15:
        return "frontal"
    elif yaw < -30:
        return "left"
    elif yaw > 30:
        return "right"
    elif abs(roll) > 20:
        return "tilted"
    else:
        return "angled"

class ClusterRequest(BaseModel):
    embeddingsPath: str
    eps: float = 0.3
    min_samples: int = 3
    faissIndexName: str

@app.post("/cluster-faces")
async def cluster_faces(req: ClusterRequest):
    try:
        # Load and normalize embeddings
        embeddings = np.load(req.embeddingsPath).astype("float32")
        normalized_embeddings = normalize(embeddings)

        # Cluster using DBSCAN
        clustering = DBSCAN(eps=req.eps, min_samples=req.min_samples, metric='cosine')
        labels = clustering.fit_predict(normalized_embeddings)

        # Save FAISS index for fast search
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(normalized_embeddings)
        os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
        index_path = os.path.join(FAISS_INDEX_DIR, f"{req.faissIndexName}.index")
        faiss.write_index(index, index_path)

        # Load metadata
        jsonl_path = req.embeddingsPath.replace(".npy", ".jsonl")
        if not os.path.exists(jsonl_path):
            return {"error": f"Metadata jsonl not found for {req.embeddingsPath}"}

        vector_map = {}
        with open(jsonl_path) as f:
            for line in f:
                item = json.loads(line)
                vector_map[item["vectorId"]] = item

        # Initialize counters
        total_faces = len(labels)
        noisy_faces_count = 0
        expanded_clusters = {}

        # Process clusters and expand with metadata
        for i, label in enumerate(labels):
            if label == -1:  # Ignore noise points
                noisy_faces_count += 1
                # continue
            
            label_str = str(label)
            if label_str not in expanded_clusters:
                expanded_clusters[label_str] = []
            
            if i in vector_map:
                expanded_clusters[label_str].append({
                    "photoId": vector_map[i]["photoId"],
                    "poseType": vector_map[i]["poseType"],
                    "detScore": vector_map[i]["detScore"],
                    "cropPath": vector_map[i]["cropPath"],
                    "sourceImagePath": vector_map[i].get("sourceImagePath", None)  # 🆕 INCLUDE SOURCE PATH
                })

        return {
            "status": "ok",
            "clusters": expanded_clusters,
            "totalFaces": total_faces,
            "noisyFacesCount": noisy_faces_count
        }

    except Exception as e:
        return structured_error("cluster_faces", "batch", str(e), traceback.format_exc())


@app.post("/cluster-faces-hdbscan")
async def cluster_faces_hdbscan(req: ClusterRequest):
    try:
        # Load and normalize embeddings
        embeddings = np.load(req.embeddingsPath).astype("float32")
        normalized_embeddings = normalize(embeddings)

        # Cluster using HDBSCAN
        clusterer = hdbscan.HDBSCAN(min_cluster_size=req.min_samples, metric='euclidean')
        labels = clusterer.fit_predict(normalized_embeddings)
        probabilities = clusterer.probabilities_

        # Save FAISS index
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(normalized_embeddings)
        os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
        index_path = os.path.join(FAISS_INDEX_DIR, f"{req.faissIndexName}.index")
        faiss.write_index(index, index_path)

        # Load metadata
        jsonl_path = req.embeddingsPath.replace(".npy", ".jsonl")
        if not os.path.exists(jsonl_path):
            return {"error": f"Metadata jsonl not found for {req.embeddingsPath}"}

        vector_map = {}
        with open(jsonl_path) as f:
            for line in f:
                item = json.loads(line)
                vector_map[item["vectorId"]] = item

        # Initialize counters and results
        total_faces = len(labels)
        noisy_faces_count = 0
        expanded_clusters = {}
        cluster_representatives = {}

        # Process clusters and expand with metadata
        for i, (label, prob) in enumerate(zip(labels, probabilities)):
            if label == -1:
                noisy_faces_count += 1
                continue  # Skip noise

            label_str = str(label)
            if label_str not in expanded_clusters:
                expanded_clusters[label_str] = []

            if i in vector_map:
                crop_path = vector_map[i]["cropPath"]
                abs_face_path = os.path.abspath(crop_path)  # 🔥 Make absolute

                photo_info = {
                    "photoId": vector_map[i]["photoId"],
                    "poseType": vector_map[i]["poseType"],
                    "detScore": vector_map[i]["detScore"],
                    "cropPath": crop_path,
                    "sourceImagePath": vector_map[i].get("sourceImagePath", None),
                    "confidence": float(prob),  # 🔥 make sure confidence is float
                    "clusterId": label_str
                }
                expanded_clusters[label_str].append(photo_info)

                # Update representative
                if label_str not in cluster_representatives or prob > cluster_representatives[label_str]["confidence"]:
                    cluster_representatives[label_str] = {
                        "photoId": vector_map[i]["photoId"],
                        "detScore": vector_map[i]["detScore"],
                        "sourceImagePath": vector_map[i].get("sourceImagePath", None),
                        "facePath": abs_face_path,  # 🔥 ADD absolute face path
                        "confidence": float(prob),
                        "clusterId": label_str
                    }

        # Build reps list
        reps = []
        for cluster_id, rep in cluster_representatives.items():
            reps.append({
                "photoId": rep["photoId"],
                "detScore": rep["detScore"],
                "sourceImagePath": rep["sourceImagePath"],
                "facePath": rep["facePath"],  # 🔥 ADD facePath in response
                "clusterId": rep["clusterId"]
            })

        return {
            "status": "ok",
            "clusters": expanded_clusters,
            "totalFaces": total_faces,
            "noisyFacesCount": noisy_faces_count,
            "rep": reps
        }

    except Exception as e:
        return structured_error("cluster_faces_hdbscan", "batch", str(e), traceback.format_exc())

class ExpandClustersRequest(BaseModel):
    clusters: Dict[str, List[int]]
    collection: str

@app.post("/expand-clusters")
async def expand_clusters(req: ExpandClustersRequest):
    try:
        jsonl_path = os.path.join("embeddings", f"{req.collection}.jsonl")
        vector_map = {}

        with open(jsonl_path) as f:
            for line in f:
                item = json.loads(line)
                vector_map[item["vectorId"]] = item

        expanded = {}
        for cluster_id, vector_ids in req.clusters.items():
            expanded[cluster_id] = [
                {
                    "photoId": vector_map[vid]["photoId"],
                    "poseType": vector_map[vid]["poseType"],
                    "detScore": vector_map[vid]["detScore"],
                    "cropPath": vector_map[vid]["cropPath"],
                    # "vectorId": vector_map[vid]["vectorId"],
                    # "faceIndex": vector_map[vid]["faceIndex"],
                    # "bbox": vector_map[vid]["bbox"],
                    # "pose": vector_map[vid]["pose"]
                }
                for vid in vector_ids if vid in vector_map
            ]

        return expanded

    except Exception as e:
        return structured_error("expand_clusters", req.collection, str(e), traceback.format_exc())

class AddVectorRequest(BaseModel):
    vector: List[float]
    meta: dict
    faissIndexName: str

@app.post("/faiss/add-vector")
async def add_vector(req: AddVectorRequest):
    try:
        vector_np = np.array(req.vector, dtype="float32").reshape(1, -1)
        index_path = os.path.join(FAISS_INDEX_DIR, f"{req.faissIndexName}.index")
        index = faiss.read_index(index_path)

        next_id = index.ntotal
        ids = np.array([next_id], dtype="int64")
        index.add_with_ids(vector_np, ids)

        faiss.write_index(index, index_path)

        req.meta["vectorId"] = int(next_id)
        req.meta["faissIndexName"] = req.faissIndexName
        os.makedirs(os.path.dirname(META_FILE), exist_ok=True)
        append_jsonl(req.meta, META_FILE)

        return {"status": "ok", "vectorId": int(next_id)}

    except Exception as e:
        return structured_error("add_vector", "na", str(e), traceback.format_exc())
    
class DinoDetectFolderRequest(BaseModel):
    folderPath: str
    collection: str

@app.post("/dino/detect-folder")
async def dino_detect_folder(req: DinoDetectFolderRequest):
    try:
        print(f"🧪 DINO detect folder called: {req.folderPath}, collection: {req.collection}")

        # Paths
        os.makedirs("embeddings", exist_ok=True)
        collection_base = os.path.join("embeddings", req.collection)
        npy_path = f"{collection_base}.npy"
        jsonl_path = f"{collection_base}.jsonl"
        index_path = f"{collection_base}.index"

        # 1. Prepare existing metadata
        existing_files = set()
        if os.path.exists(jsonl_path):
            with open(jsonl_path, "r") as f:
                for line in f:
                    item = json.loads(line)
                    existing_files.add(os.path.basename(item["sourceImagePath"]).lower())

        print(f"Found {len(existing_files)} already indexed files.")

        # 2. Load all image paths
        if not os.path.exists(req.folderPath):
            return {"error": f"Folder {req.folderPath} does not exist"}

        all_images = [
            os.path.join(req.folderPath, f)
            for f in os.listdir(req.folderPath)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]

        new_images = [
            img for img in all_images
            if os.path.basename(img).lower() not in existing_files
        ]

        print(f"{len(new_images)} new images to process.")

        if not new_images:
            return {"status": "no new images to process"}

        # 3. Embed using DINO
        import torch
        import torchvision.transforms as T
        from torchvision.models import vit_b_16
        from PIL import Image

        model = vit_b_16(weights="IMAGENET1K_V1")
        model.eval()
        model = model.cuda() if torch.cuda.is_available() else model

        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        embeddings = []
        metadata = []

        for img_path in new_images:
            try:
                img = Image.open(img_path).convert("RGB")
                img_tensor = transform(img).unsqueeze(0)
                img_tensor = img_tensor.cuda() if torch.cuda.is_available() else img_tensor

                with torch.no_grad():
                    feature = model(img_tensor).squeeze(0).cpu().numpy()

                embeddings.append(feature)
                metadata.append({
                    "sourceImagePath": img_path,
                })

            except Exception as e:
                print(f"❌ Error processing {img_path}: {e}")

        embeddings = np.vstack(embeddings).astype("float32")

        # 4. Append to existing npy if any
        if os.path.exists(npy_path):
            existing_embeddings = np.load(npy_path)
            all_embeddings = np.concatenate([existing_embeddings, embeddings], axis=0)
        else:
            all_embeddings = embeddings

        np.save(npy_path, all_embeddings)
        print(f"💾 Saved {all_embeddings.shape[0]} embeddings → {npy_path}")

        # 5. Write or append to JSONL
        with open(jsonl_path, "a") as f:
            for item in metadata:
                f.write(json.dumps(item) + "\n")

        print(f"📝 Appended {len(metadata)} entries to {jsonl_path}")

        # 6. Build FAISS index
        index = faiss.IndexFlatL2(all_embeddings.shape[1])
        index.add(all_embeddings)
        faiss.write_index(index, index_path)
        print(f"📦 Saved FAISS index → {index_path}")

        return {
            "status": "ok",
            "filesProcessed": len(new_images),
            "indexSize": index.ntotal,
            "collection": req.collection
        }

    except Exception as e:
        return structured_error("dino_detect_folder", "batch", str(e), traceback.format_exc())

    
class CreateDinoFaissIndexRequest(BaseModel):
    embeddingsPath: str
    faissIndexName: str
    normalize: bool = True  # Optional: normalize for cosine search

@app.post("/dino/create-faiss-index")
async def create_dino_faiss_index(req: CreateDinoFaissIndexRequest):
    try:
        print(f"🧪 Creating DINO FAISS index: {req.embeddingsPath} → {req.faissIndexName}")

        if not os.path.exists(req.embeddingsPath):
            return {"error": f"Embeddings file {req.embeddingsPath} not found"}

        embeddings = np.load(req.embeddingsPath).astype("float32")
        
        if req.normalize:
            faiss.normalize_L2(embeddings)
            index = faiss.IndexFlatIP(embeddings.shape[1])  # Cosine similarity
        else:
            index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 distance

        index.add(embeddings)
        os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
        index_path = os.path.join(FAISS_INDEX_DIR, f"{req.faissIndexName}.index")
        faiss.write_index(index, index_path)

        print(f"📦 Saved FAISS index → {index_path}")

        return {
            "status": "ok",
            "indexPath": index_path,
            "vectorsAdded": embeddings.shape[0]
        }

    except Exception as e:
        return structured_error("create_dino_faiss_index", "batch", str(e), traceback.format_exc())


class HDBScanRequest(BaseModel):
    embeddingsPath: str
    metadataPath: str
    min_cluster_size: int = Form(3)

@app.post("/hdbscan-cluster")
async def hdbscan_cluster(req: HDBScanRequest):
    try:
        # Load embeddings
        embeddings = np.load(req.embeddingsPath).astype("float32")
        print(f"✅ Loaded embeddings of shape: {embeddings.shape}")

        # Normalize
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        start = time.time()
        print(f"🚀 Starting HDBSCAN clustering with min_cluster_size={req.min_cluster_size}...")

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=req.min_cluster_size,
            min_samples=None,
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True
        )
        labels = clusterer.fit_predict(embeddings)

        end = time.time()
        print(f"✅ Clustering completed in {(end - start) / 60:.2f} minutes")

        # Save labels
        output_dir = "cluster_results"
        os.makedirs(output_dir, exist_ok=True)

        filename = os.path.basename(req.embeddingsPath)
        filename_no_ext = os.path.splitext(filename)[0]
        output_path = os.path.join(output_dir, f"labels_{filename_no_ext}.pkl")
        joblib.dump(labels, output_path)

        # Count clusters and noise
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)

        # Load metadata (for adding cropPath, detScore etc)
        jsonl_path = req.metadataPath  # 🆕 add metadataPath to your HDBScanRequest
        metadata = []
        with open(jsonl_path) as f:
            for line in f:
                metadata.append(json.loads(line))

        # Build clusters with metadata
        clusters = {}
        for idx, label in enumerate(labels):
            if label == -1:
                continue  # Skip noise if you want

            if label not in clusters:
                clusters[label] = []

            face_meta = metadata[idx]  # Assume alignment is correct
            det_score = face_meta.get("detScore")

            # Skip faces with detScore < 0.7
            if det_score is None or float(det_score) < 0.7:
                continue  # ❌ skip this face
            clusters[label].append({
                "index": int(idx),  # Force int
                "cropPath": face_meta.get("cropPath"),
                "sourceImagePath": face_meta.get("sourceImagePath"),
                "detScore": float(face_meta.get("detScore")) if face_meta.get("detScore") is not None else None  # Force float
            })


        # Final structured result
        cluster_details = []
        for cluster_id, faces in clusters.items():
            cluster_details.append({
                "clusterId": int(cluster_id),
                "faces": faces
            })

        cluster_details = sorted(cluster_details, key=lambda x: x["clusterId"])

        return {
            "message": "Clustering completed",
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "labels_file": output_path,
            "clusters": cluster_details   # 🆕 Include all faces + scores in response!
        }

    except Exception as e:
        return {"error": str(e)}


class EmbedRequest(BaseModel):
    image_path: str
    npy_path: str
    jsonl_path: str

class FolderRequest(BaseModel):
    folder_path: str
    npy_path: str
    jsonl_path: str

def embed_image(image_path: str) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = clip_model.encode_image(image_input)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding.cpu().numpy()

@app.post("/clip/embed-image/")
async def embed_image_and_save(req: EmbedRequest):
    embedding = embed_image(req.image_path)

    # Create or append to .npy file
    if not os.path.exists(req.npy_path):
        np.save(req.npy_path, embedding)
        vector_id = 0
    else:
        existing = np.load(req.npy_path)
        combined = np.vstack([existing, embedding])
        np.save(req.npy_path, combined)
        vector_id = existing.shape[0]  # new vector id is previous size

    # Create or append to .jsonl file
    jsonl_entry = {
        "vector_id": vector_id,
        "filepath": req.image_path
    }
    with open(req.jsonl_path, "a") as f:
        f.write(json.dumps(jsonl_entry) + "\n")

    return {"message": f"Embedding for {req.image_path} added to {req.npy_path} and {req.jsonl_path}"}

@app.post("/clip/embed-folder/")
async def embed_folder(req: FolderRequest):
    files = [f for f in os.listdir(req.folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    files.sort()  # Optional

    for file in files:
        full_path = os.path.join(req.folder_path, file)
        await embed_image_and_save(EmbedRequest(
            image_path=full_path,
            npy_path=req.npy_path,
            jsonl_path=req.jsonl_path
        ))

    return {"message": f"Embedded {len(files)} images from {req.folder_path} into {req.npy_path} and {req.jsonl_path}"}


# Request model for clustering
class ClusterRequest(BaseModel):
    embeddingsPath: str  # path to .npy
    eps: float = 0.3     # DBSCAN eps default
    min_samples: int = 3 # DBSCAN min_samples default
    faissIndexName: str  # name to save FAISS index

FAISS_INDEX_DIR = "./faiss_indexes"  # You can adjust this path

def structured_error(stage, photoId, message, trace):
    return {
        "error": True,
        "stage": stage,
        "photoId": photoId,
        "message": message,
        "trace": trace
    }


# Request model
class ClusterRequest(BaseModel):
    embeddingsPath: str
    eps: float = 0.12
    min_samples: int = 2
    faissIndexName: str

FAISS_INDEX_DIR = "./faiss_indexes"  # Adjust if needed

def structured_error(stage, photoId, message, trace):
    return {
        "error": True,
        "stage": stage,
        "photoId": photoId,
        "message": message,
        "trace": trace
    }

@app.post("/clip/cluster/")
async def cluster_clip_embeddings(req: ClusterRequest):
    try:
        # Load and normalize embeddings
        embeddings = np.load(req.embeddingsPath).astype("float32")
        normalized_embeddings = normalize(embeddings)

        # Cluster using DBSCAN
        clustering = DBSCAN(eps=req.eps, min_samples=req.min_samples, metric='cosine')
        labels = clustering.fit_predict(normalized_embeddings)

        # Save FAISS index
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(normalized_embeddings)
        os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
        index_path = os.path.join(FAISS_INDEX_DIR, f"{req.faissIndexName}.index")
        faiss.write_index(index, index_path)

        # Load metadata from .jsonl
        jsonl_path = req.embeddingsPath.replace(".npy", ".jsonl")
        if not os.path.exists(jsonl_path):
            return {"error": f"Metadata jsonl not found for {req.embeddingsPath}"}

        vector_map = {}
        with open(jsonl_path) as f:
            for line in f:
                item = json.loads(line)
                vector_map[item["vector_id"]] = item

        # Create expanded cluster structure
        expanded_clusters = {}
        total_items = len(labels)

        for i, label in enumerate(labels):
            label_str = str(label)
            if label_str not in expanded_clusters:
                expanded_clusters[label_str] = []
            if i in vector_map:
                expanded_clusters[label_str].append({
                    "filepath": vector_map[i]["filepath"]
                })

        # Save clusters result as .json
        clusters_output = {
            "status": "ok",
            "clusters": expanded_clusters,
            "totalItems": total_items
        }
        save_json_path = req.embeddingsPath.replace(".npy", "_clusters.json")
        with open(save_json_path, "w") as f:
            json.dump(clusters_output, f)

        # Return normal API response + path
        return {
            "status": "ok",
            "clustersSavedAt": save_json_path,
            "totalItems": total_items
        }

    except Exception as e:
        return structured_error("cluster_clip_embeddings", "batch", str(e), traceback.format_exc())


# Your request model
class HDBSCANClusterRequest(BaseModel):
    embeddingsPath: str   # path to .npy
    faissIndexName: str   # name to save FAISS index
    min_cluster_size: int = 2   # min samples to form a cluster (default = 5)
    cluster_selection_epsilon: float=0.5


@app.post("/clip/hdbscan-cluster/")
async def cluster_clip_embeddings_hdbscan(req: HDBSCANClusterRequest):
    try:
        # Load and normalize embeddings
        embeddings = np.load(req.embeddingsPath).astype("float32")
        normalized_embeddings = normalize(embeddings)

        # Cluster using HDBSCAN
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=req.min_cluster_size,
            min_samples=1,
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True,
            cluster_selection_epsilon=req.cluster_selection_epsilon
        )
        labels = clusterer.fit_predict(normalized_embeddings)

        # Save FAISS index
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(normalized_embeddings)
        os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
        index_path = os.path.join(FAISS_INDEX_DIR, f"{req.faissIndexName}.index")
        faiss.write_index(index, index_path)

        # Load metadata
        jsonl_path = req.embeddingsPath.replace(".npy", ".jsonl")
        if not os.path.exists(jsonl_path):
            return {"error": f"Metadata jsonl not found for {req.embeddingsPath}"}

        vector_map = {}
        with open(jsonl_path) as f:
            for line in f:
                item = json.loads(line)
                vector_map[item["vector_id"]] = item

        # Expand clusters with filepaths
        expanded_clusters = {}
        total_items = len(labels)

        for i, label in enumerate(labels):
            label_str = str(label)
            if label_str not in expanded_clusters:
                expanded_clusters[label_str] = []
            if i in vector_map:
                expanded_clusters[label_str].append({
                    "filepath": vector_map[i]["filepath"]
                })

        # Save clusters to JSON
        clusters_output = {
            "status": "ok",
            "clusters": expanded_clusters,
            "totalItems": total_items
        }
        save_json_path = req.embeddingsPath.replace(".npy", "_clusters_hdbscan.json")
        with open(save_json_path, "w") as f:
            json.dump(clusters_output, f)

        # Return basic response
        return {
            "status": "ok",
            "clustersSavedAt": save_json_path,
            "totalItems": total_items,
            "n_clusters": len(set(labels)) - (1 if -1 in labels else 0),
            "n_noise": list(labels).count(-1)
        }

    except Exception as e:
        return structured_error("cluster_clip_embeddings_hdbscan", "batch", str(e), traceback.format_exc())


# Start server locally
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
