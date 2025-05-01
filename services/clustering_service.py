import os
import numpy as np
import traceback
from sklearn.cluster import DBSCAN
import hdbscan
from config import get_embeddings_dir, get_metadata_path, get_faces_metadata_path
from utils.jsonl_utils import load_jsonl
from utils.error_utils import log_error
import logging
from PIL import Image, ImageOps
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

VALID_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png'}  # Valid image extensions


# --- Common loading ---
def is_valid_image_filename(filename):
        """Check if filename has a valid image extension."""
        return filename.lower().endswith(tuple(VALID_IMAGE_EXTENSIONS))

def load_embeddings_and_metadata(partner: str, model: str):
    embeddings_dir = get_embeddings_dir(partner)
    if model == "clip":
        npy_path = os.path.join(embeddings_dir, f"{partner}_clip.npy")
        metadata_path = get_metadata_path(partner)
    elif model == "dino":
        npy_path = os.path.join(embeddings_dir, f"{partner}_dino.npy")
        metadata_path = get_metadata_path(partner)
    else:
        raise ValueError("Unsupported model")

    embeddings = np.load(npy_path)
    metadata = load_jsonl(metadata_path)

    vectorid_to_metadata = {}
    for entry in metadata:
        if model == "clip":
            vector_id = entry.get("clip", {}).get("vectorId")
        elif model == "dino":
            vector_id = entry.get("dino", {}).get("vectorId")
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

# --- DBSCAN and HDBSCAN runners ---
def run_dbscan(embeddings, eps=0.5, min_samples=3, metric="euclidean"):
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric=metric).fit(embeddings)
    return clustering.labels_

def run_hdbscan(embeddings, min_cluster_size=2, metric="euclidean"):
    clusterer = hdbscan.HDBSCAN(
        # min_cluster_size=min_cluster_size,
        # min_samples=5, for insightface
        min_samples=1,
        metric=metric,
        cluster_selection_method='eom',
        prediction_data=True,
        cluster_selection_epsilon=0,
        min_cluster_size=2
        # cluster_selection_epsilon=0.5 for insightface
    )
    # Fit and predict cluster labels
    labels = clusterer.fit_predict(embeddings)

    # Get membership probabilities
    probabilities = clusterer.probabilities_

    return labels, probabilities

def build_clusters(labels, metadata, probabilities, model: str, partner: str):
    """
    Build clusters from HDBSCAN labels, metadata, and probabilities, and save response as JSON.
    
    Args:
        labels (np.ndarray): Cluster labels from HDBSCAN (-1 for noise).
        metadata (list): List of dictionaries with face metadata (e.g., faceId, detScore, bbox).
        probabilities (np.ndarray): Membership probabilities for each point's cluster.
        model (str): Detection model ('insightface' or other).
        partner (str): Identifier for the data source, used in directory paths.
    
    Returns:
        dict: Clustering results with totalClusters, totalItems, reps, items, threshold, and summary.
    """
    DET_SCORE_THRESHOLD = 0.7  # Recommended threshold for InsightFace
    VALID_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png'}  # Valid image extensions
    BASE_DIR = "/Users/sumit/Documents/ai_analysis"
    
    # Validate partner (basic check for safe directory name)
    if not partner or not all(c.isalnum() or c in '-_' for c in partner):
        logger.error(f"Invalid partner identifier: {partner}")
        raise ValueError("Partner must be alphanumeric with hyphens or underscores")
    
    # Construct directories
    PHOTOS_DIR = os.path.join(BASE_DIR, partner, "photos")
    FACES_REP_DIR = os.path.join(BASE_DIR, partner, "faces_rep")
    
    # Create faces_rep directory if it doesn't exist
    os.makedirs(FACES_REP_DIR, exist_ok=True)

    
    def crop_and_save_rep(source_path, bbox, face_id):
        """Crop image using bbox and save to faces_rep, return cropPath."""
        try:
            if not source_path or not os.path.exists(source_path):
                logger.error(f"Source image not found: {source_path}")
                return None
            if not is_valid_image_filename(source_path):
                logger.error(f"Invalid source image extension: {source_path}")
                return None
            if not bbox or not isinstance(bbox, list) or len(bbox) != 4:
                logger.error(f"Invalid bbox for faceId={face_id}: {bbox}")
                return None

            # Open image and apply EXIF orientation
            with Image.open(source_path) as img:
                img = img.convert('RGB')
                raw_width, raw_height = img.size
                logger.info(f"Raw image dimensions for faceId={face_id}: {raw_width}x{raw_height}")

                # Apply EXIF orientation (e.g., rotate 90° clockwise for Orientation 6)
                img = ImageOps.exif_transpose(img)
                img_width, img_height = img.size
                logger.info(f"Rotated image dimensions for faceId={face_id}: {img_width}x{img_height}")

                # Parse bbox as [x_min, y_min, x_max, y_max]
                x1, y1, x2, y2 = map(float, bbox)  # Use float to handle decimals
                # logger.info(f"Original bbox for faceId={face_id}: [{x1}, {y1}, {x2}, {y2}]")

                # Ensure x_min < x_max and y_min < y_max
                x_min, x_max = min(x1, x2), max(x1, x2)
                y_min, y_max = min(y1, y2), max(y1, y2)

                # Round to integers for cropping
                x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])
                logger.info(f"Adjusted bbox for faceId={face_id}: [{x_min}, {y_min}, {x_max}, {y_max}]")

                # Validate adjusted bbox
                if x_max <= x_min or y_max <= y_min:
                    logger.error(f"Invalid adjusted bbox dimensions for faceId={face_id}: [{x_min}, {y_min}, {x_max}, {y_max}]")
                    return None
                if x_min < 0 or y_min < 0 or x_max > img_width or y_max > img_height:
                    logger.error(f"Bbox out of image bounds for faceId={face_id}: [{x_min}, {y_min}, {x_max}, {y_max}]")
                    return None

                # Crop image
                cropped_img = img.crop((x_min, y_min, x_max, y_max))

                # Save cropped image
                crop_filename = f"{face_id}.jpg"
                crop_path = os.path.join(FACES_REP_DIR, crop_filename)
                cropped_img.save(crop_path, 'JPEG', quality=95)
                logger.info(f"Saved cropped image: {crop_path}")
                return f"/{partner}/faces_rep/{crop_filename}"
        except Exception as e:
            logger.error(f"Failed to crop/save for faceId={face_id}: {str(e)}")
            return None

    clusters = {}
    cluster_summary = []  # To store summary for each cluster

    # Build clusters with filtered entries
    for idx, (label, prob) in enumerate(zip(labels, probabilities)):
        label = int(label)
        entry = metadata[idx]
        
        # Skip low-quality faces for InsightFace
        if model == "insightface" and entry.get("detScore", 0.0) < DET_SCORE_THRESHOLD:
            continue

        confidence = float(prob) if label != -1 else 0.0  # Membership probability as confidence

        # Extract filenames
        crop_filename = os.path.basename(entry.get('cropPath', '')) if entry.get('cropPath') else ''
        source_filename = os.path.basename(entry.get('sourceImagePath', '')) if entry.get('sourceImagePath') else ''
        
        # Construct and validate paths
        face_path = f"/{partner}/faces/{crop_filename}" if crop_filename and is_valid_image_filename(crop_filename) else None
        image_path = f"/{partner}/photos/{source_filename}" if source_filename and is_valid_image_filename(source_filename) else None

        # Log for debugging
        logger.info(f"Processing faceId: {entry.get('faceId')}, sourceImagePath: {entry.get('sourceImagePath')}, imagePath: {image_path}")

        if model == "insightface":
            item = {
                "clusterId": label if label != -1 else None,
                "faceId": str(entry.get("faceId", "")) or "",
                "photoId": str(entry.get("photoId", "")) or "",
                "facePath": face_path,
                "imagePath": image_path,
                "detScore": float(entry.get("detScore", 0.0)),
                "aestheticScore": None,
                "bbox": entry.get("bbox", []),
                "noise": label == -1,
                "confidence": confidence
            }
        else:
            item = {
                "clusterId": label if label != -1 else "Noise",
                "faceId": None,
                "photoId": str(entry.get("id", "")) or "",
                "facePath": None,
                "imagePath": image_path,
                "detScore": None,
                "aestheticScore": float(entry.get("aestheticScore", 0.0)),
                "bbox": entry.get("bbox", []),  # Include bbox for consistency
                "noise": label == -1,
                "confidence": confidence
            }

        clusters.setdefault(label if label != -1 else "noise", []).append(item)

    all_items = []
    reps = []

    # Process clusters and build summary
    for cluster_id, faces in clusters.items():
        total_faces = len(faces)
        faces_above_threshold = sum(1 for face in faces if face.get("detScore", 0.0) >= DET_SCORE_THRESHOLD) if model == "insightface" else total_faces

        # Add to summary
        cluster_summary.append({
            "clusterId": cluster_id if cluster_id != "noise" else None,
            "totalFaces": total_faces,
            "facesAboveThreshold": faces_above_threshold
        })

        if cluster_id == "noise" and model == "insightface":
            # Process noise cluster
            rep = max(faces, key=lambda x: x["detScore"] or 0)
            if rep["detScore"] >= DET_SCORE_THRESHOLD:
                rep = rep.copy()
                source_path = rep.get("imagePath", "").replace(f"/{partner}/images/", f"{PHOTOS_DIR}/")
                rep["cropPath"] = crop_and_save_rep(source_path, rep.get("bbox"), rep.get("faceId"))
                reps.append(rep)
            cropped_noise_faces = []
            for face in faces:
                if face["detScore"] >= DET_SCORE_THRESHOLD:
                    face = face.copy()
                    source_path = face.get("imagePath", "").replace(f"/{partner}/images/", f"{PHOTOS_DIR}/")
                    face["cropPath"] = crop_and_save_rep(source_path, face.get("bbox"), face.get("faceId"))
                cropped_noise_faces.append(face)
            all_items.extend(sorted(cropped_noise_faces, key=lambda x: x.get("confidence", 0), reverse=True))
        else:
            # Process non-noise clusters
            if model == "insightface":
                rep = max(faces, key=lambda x: x["detScore"] or 0)
                faces_sorted = sorted(faces, key=lambda x: x.get("confidence", 0), reverse=True)
            else:
                rep = max(faces, key=lambda x: x["aestheticScore"] or 0)
                faces_sorted = sorted(faces, key=lambda x: x.get("confidence", 0), reverse=True)

            rep = rep.copy()
            source_path = rep.get("imagePath", "").replace(f"/{partner}/images/", f"{PHOTOS_DIR}/")
            rep["cropPath"] = crop_and_save_rep(source_path, rep.get("bbox"), rep.get("faceId"))
            reps.append(rep)
            all_items.extend(faces_sorted)

    # Prepare response
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    response = {
        "clusterVersion": timestamp,
        "totalClusters": len([k for k in clusters.keys() if k != "noise"]),
        "totalItems": len(all_items),
        "reps": reps,
        "items": all_items,
        "threshold": DET_SCORE_THRESHOLD if model == "insightface" else None,
        "summary": cluster_summary if model == "insightface" else []
    }

    # Save response as JSON with timestamp
    try:
        json_filename = f"cluster_faces.json"
        json_path = os.path.join(FACES_REP_DIR, json_filename)
        with open(json_path, 'w') as f:
            json.dump(response, f, indent=2)
        logger.info(f"Saved cluster response: {json_path}")
    except Exception as e:
        logger.error(f"Failed to save cluster response to {json_path}: {str(e)}")

    return response

# --- Main endpoints ---
async def cluster_embeddings_and_build_response(req):
    try:
        embeddings_dir = get_embeddings_dir(req.partner)
        if req.model == "clip" or req.model == "dino":
            embeddings, metadata = load_embeddings_and_metadata(req.partner, req.model)
        else:
            raise ValueError("Use separate endpoint for InsightFace!")

        # metric = req.distance if req.distance else 'euclidean'
        # metric = "cosine"
        # if metric == "cosine":
        metric = "euclidean"
        norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.clip(norm, 1e-10, None)

        if req.method == "hdbscan":
            labels, probabilities = run_hdbscan(embeddings, min_cluster_size=req.minClusterSize or 2,metric=metric)
            unique_labels = np.unique(labels)
            print(f"Unique labels: {unique_labels}, Number of clusters: {len(unique_labels[unique_labels >= 0])}")
        elif req.method == "dbscan":
            labels = run_dbscan(embeddings, eps=req.eps or 0.5, min_samples=2, metric=metric)
        else:
            raise ValueError("Unsupported clustering method")

        if req.model in ["clip", "dino"]:
            return build_clusters_clip_dino(labels, metadata, probabilities, req.model, req.partner)
        else:
            return build_clusters(labels, metadata, probabilities, req.model, req.partner)

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

async def cluster_faces_insightface(req):
    try:
        embeddings_dir = get_embeddings_dir(req.partner)
        npy_path = os.path.join(embeddings_dir, f"{req.partner}_insightface.npy")
        metadata_path = get_faces_metadata_path(req.partner)
        embeddings = np.load(npy_path)
        metadata = load_jsonl(metadata_path)
        metric = req.distance if req.distance else 'euclidean'

        if metric == "cosine":
            norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.clip(norm, 1e-10, None)

        if req.method == "hdbscan":
            labels, membership_strength = run_hdbscan(embeddings, min_cluster_size=req.minClusterSize or 2, metric=metric)
        elif req.method == "dbscan":
            labels = run_dbscan(embeddings, eps=req.eps or 0.5, min_samples=2, metric=metric)
        else:
            raise ValueError("Unsupported clustering method")

        # Sort metadata properly for insightface
        metadata = sorted(metadata, key=lambda x: x.get("vectorId", 0))

        return build_clusters(labels, metadata, membership_strength, "insightface", req.partner)

    except Exception as e:
        log_error(
            photoId=None,
            faceId=None,
            step="cluster-insightface",
            errorMessage=str(e),
            traceback=traceback.format_exc(),
            modelType="insightface",
            path=None,
            partner=req.partner
        )
        raise


def build_clusters_clip_dino(labels, metadata, probabilities, model: str, partner: str):
    """
    Build clusters for CLIP/DINO from HDBSCAN labels, metadata, and probabilities.
    """
    BASE_DIR = "/Users/sumit/Documents/ai_analysis"
    
    if not partner or not all(c.isalnum() or c in '-_' for c in partner):
        logger.error(f"Invalid partner identifier: {partner}")
        raise ValueError("Partner must be alphanumeric with hyphens or underscores")
    
    PHOTOS_DIR = os.path.join(BASE_DIR, partner, "photos")
    CLIP_REP_DIR = os.path.join(BASE_DIR, partner)
    os.makedirs(CLIP_REP_DIR, exist_ok=True)

    clusters = {}
    cluster_summary = []

    for idx, (label, prob) in enumerate(zip(labels, probabilities)):
        label = int(label)
        entry = metadata[idx]
        confidence = float(prob) if label != -1 else 0.0

        source_filename = os.path.basename(entry.get('fileName', '')) if entry.get('fileName') else ''
        image_path = f"/{partner}/photos/{source_filename}" if source_filename and is_valid_image_filename(source_filename) else None

        item = {
            "clusterId": label if label != -1 else "Noise",
            "faceId": None,
            "photoId": str(entry.get("id", "")) or "",
            "facePath": None,
            "imagePath": image_path,
            "detScore": None,
            "aestheticScore": float(entry.get("aestheticScore", 0.0)),
            "bbox": entry.get("bbox", []),
            "noise": label == -1,
            "confidence": confidence
        }

        clusters.setdefault(label if label != -1 else "noise", []).append(item)

    all_items = []
    reps = []

    for cluster_id, items in clusters.items():
        total_items = len(items)
        cluster_summary.append({
            "clusterId": cluster_id if cluster_id != "noise" else None,
            "totalFaces": total_items,
            "facesAboveThreshold": total_items  # No threshold for CLIP/DINO
        })

        if cluster_id == "noise":
            rep = max(items, key=lambda x: x["aestheticScore"] or 0)
            rep = rep.copy()
            image_path = rep.get("imagePath")
            if image_path and isinstance(image_path, str) and is_valid_image_filename(image_path):
                reps.append(rep)
            else:
                logger.warning(f"Skipping noise cluster rep due to invalid imagePath: {image_path}")
            all_items.extend(sorted(items, key=lambda x: x.get("confidence", 0), reverse=True))
        else:
            rep = max(items, key=lambda x: x["aestheticScore"] or 0)
            faces_sorted = sorted(items, key=lambda x: x.get("confidence", 0), reverse=True)
            rep = rep.copy()
            image_path = rep.get("imagePath")
            if image_path and isinstance(image_path, str) and is_valid_image_filename(image_path):
                reps.append(rep)
            else:
                logger.warning(f"Skipping cluster {cluster_id} rep due to invalid imagePath: {image_path}")
            all_items.extend(faces_sorted)

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    response = {
        "clusterVersion": timestamp,
        "totalClusters": len([k for k in clusters.keys() if k != "noise"]),
        "totalItems": len(all_items),
        "reps": reps,
        "items": all_items,
        "threshold": None,
        "summary": cluster_summary
    }

    try:
        json_filename = f"cluster_{model}.json"
        json_path = os.path.join(CLIP_REP_DIR, json_filename)
        with open(json_path, 'w') as f:
            json.dump(response, f, indent=2)
        logger.info(f"Saved cluster response: {json_path}")
    except Exception as e:
        logger.error(f"Failed to save cluster response to {json_path}: {str(e)}")

    return response