# services/clustering_service.py

import os
import numpy as np
import traceback
from sklearn.cluster import DBSCAN
import hdbscan
from config import get_embeddings_dir, get_metadata_path, get_faces_metadata_path
from utils.jsonl_utils import load_jsonl
from utils.error_utils import log_error

def load_embeddings_and_metadata(partner: str, model: str):
    embeddings_dir = get_embeddings_dir(partner)
    if model == "clip":
        npy_path = os.path.join(embeddings_dir, f"{partner}_clip.npy")
        metadata_path = get_metadata_path(partner)
    elif model == "dino":
        npy_path = os.path.join(embeddings_dir, f"{partner}_dino.npy")
        metadata_path = get_metadata_path(partner)
    elif model == "insightface":
        npy_path = os.path.join(embeddings_dir, f"{partner}_insightface.npy")
        metadata_path = get_faces_metadata_path(partner)
    else:
        raise ValueError("Unsupported model")

    embeddings = np.load(npy_path)
    metadata = load_jsonl(metadata_path)

    # 🔥 NEW PART: reorder metadata correctly based on vectorId
    vectorid_to_metadata = {}

    for entry in metadata:
        if model == "clip":
            vector_id = entry.get("clip", {}).get("vectorId")
        elif model == "dino":
            vector_id = entry.get("dino", {}).get("vectorId")
        elif model == "insightface":
            vector_id = entry.get("vectorId")
        else:
            vector_id = None

        if vector_id is not None:
            vectorid_to_metadata[vector_id] = entry

    ordered_metadata = []
    for idx in range(embeddings.shape[0]):
        entry = vectorid_to_metadata.get(idx)
        if entry is None:
            raise Exception(f"Missing metadata for vector index {idx}")
        ordered_metadata.append(entry)

    return embeddings, ordered_metadata

def run_dbscan(embeddings, eps=0.5, min_samples=3):
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean").fit(embeddings)
    return clustering.labels_

def run_hdbscan(embeddings, min_cluster_size=2):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=1,metric='euclidean',
    cluster_selection_method='eom',
    prediction_data=True,
    cluster_selection_epsilon=0.0000001)
    labels = clusterer.fit_predict(embeddings)
    return labels

def build_clusters(labels, metadata, model: str, partner: str):
    clusters = {}

    for idx, label in enumerate(labels):
        label = int(label)

        entry = metadata[idx]
        print("label is",label, "idx is",idx,"entry is",entry)
        item = {}

        if model == "insightface":
            item = {
                "clusterId": label if label != -1 else None,
                "faceId": str(entry.get("faceId")),
                "photoId": str(entry.get("photoId")),
                "facePath": entry.get("cropPath"),
                "imagePath": f"/images/{entry.get('fileName')}",  # 👈 FIXED
                "detScore": float(entry.get("detScore", 0.0)),
                "aestheticScore": None,
                "noise": label == -1
            }
        else:
            item = {
                "clusterId": label if label != -1 else "Noise",
                "faceId": None,
                "photoId": str(entry.get("id")),
                "facePath": None,
                "imagePath": f"/images/{entry.get('fileName')}",  # 👈 FIXED
                "detScore": None,
                "aestheticScore": float(entry.get("aestheticScore", 0.0)),
                "noise": label == -1
            }

        if label != -1:
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(item)
        else:
            if "noise" not in clusters:
                clusters["noise"] = []
            clusters["noise"].append(item)

    all_items = []
    reps = []

    for cluster_id, faces in clusters.items():
        if cluster_id == "noise":
            continue  # no reps for noise
        if model == "insightface":
            rep = max(faces, key=lambda x: x["detScore"] or 0)
            faces_sorted = sorted(faces, key=lambda x: x["detScore"] or 0, reverse=True)
        else:
            rep = max(faces, key=lambda x: x["aestheticScore"] or 0)
            faces_sorted = sorted(faces, key=lambda x: x["aestheticScore"] or 0, reverse=True)

        reps.append(rep)
        all_items.extend(faces_sorted)

    if "noise" in clusters:
        all_items.extend(clusters["noise"])

    reps = sorted(reps, key=lambda x: x["detScore"] or x["aestheticScore"] or 0, reverse=True)

    return {
        "totalClusters": len([k for k in clusters.keys() if k != "noise"]),
        "totalItems": len(all_items),
        "reps": reps,
        "items": all_items
    }

def cluster_faces_dbscan(partner: str, model: str, eps=0.5, min_samples=2):
    try:
        embeddings, metadata = load_embeddings_and_metadata(partner, model)
        labels = run_dbscan(embeddings, eps, min_samples)
        return build_clusters(labels, metadata, model,partner)
    except Exception as e:
        log_error(
            photoId=None,
            faceId=None,
            step="cluster-dbscan",
            errorMessage=str(e),
            traceback=traceback.format_exc(),
            modelType=model,
            path=None,
            partner=partner
        )
        raise

def cluster_faces_hdbscan(partner: str, model: str, min_cluster_size=5):
    try:
        embeddings, metadata = load_embeddings_and_metadata(partner, model)
        labels = run_hdbscan(embeddings, min_cluster_size)
        print("labels are",labels)
        return build_clusters(labels, metadata, model,partner)
    except Exception as e:
        log_error(
            photoId=None,
            faceId=None,
            step="cluster-hdbscan",
            errorMessage=str(e),
            traceback=traceback.format_exc(),
            modelType=model,
            path=None,
            partner=partner
        )
        raise


# services/clustering_service.py (continued)

async def cluster_embeddings_and_build_response(req):
    print("IN cluster_embeddings_and_build_response",req.method)
    try:
        embeddings_dir = get_embeddings_dir(req.partner)

        if req.model == "clip":
            npy_path = os.path.join(embeddings_dir, f"{req.partner}_clip.npy")
            metadata_path = get_metadata_path(req.partner)
        elif req.model == "dino":
            npy_path = os.path.join(embeddings_dir, f"{req.partner}_dino.npy")
            metadata_path = get_metadata_path(req.partner)
        elif req.model == "insightface":
            npy_path = os.path.join(embeddings_dir, f"{req.partner}_insightface.npy")
            metadata_path = get_faces_metadata_path(req.partner)
        else:
            raise ValueError("Unsupported model type")

        embeddings = np.load(npy_path)
        metadata = load_jsonl(metadata_path)

        # Normalize if using cosine distance
        metric = req.distance if req.distance else 'euclidean'
        if metric == "cosine":
            norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.clip(norm, 1e-10, None)

        # Sort metadata properly
        if req.model in ["clip", "dino"]:
            metadata = sorted(metadata, key=lambda x: x.get(req.model, {}).get("vectorId", 0))

        # Run clustering
        if req.method == "hdbscan":
            print("in --- hdbscan")
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=req.minClusterSize or 2,
                min_samples=1,
                metric='euclidean',
                cluster_selection_method='eom',
                prediction_data=True,
                cluster_selection_epsilon=1e-7
            )
            labels = clusterer.fit_predict(embeddings)

        elif req.method == "dbscan":
            print("in --- DBSCAN")
            clusterer = DBSCAN(
                eps=req.eps or 0.5,
                min_samples=2,
                metric=metric
            )
            labels = clusterer.fit_predict(embeddings)

        else:
            raise ValueError("Unsupported clustering method")

        return build_clusters(labels, metadata, req.model, req.partner)

    except Exception as e:
        log_error(
            photoId=None,
            faceId=None,
            step="cluster-service",
            errorMessage=str(e),
            traceback=traceback.format_exc(),
            modelType=req.model,
            path=None,
            partner=req.partner
        )
        raise

 