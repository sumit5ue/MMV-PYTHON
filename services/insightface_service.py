# services/insightface_service.py

import numpy as np
import os
import traceback
import uuid
import cv2
from PIL import Image
from config import get_embeddings_dir, get_metadata_path, get_photos_dir, get_faces_dir
from utils.jsonl_utils import load_jsonl, save_jsonl
from utils.error_utils import log_error
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
from db.sqlite_faces import save_faces_to_db
import json

from uuid import UUID


# Step 1: Load face detection + embeddings (buffalo_l)
face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=0)

from PIL import Image, ExifTags


def load_image_with_exif_rotation(path):
    image = Image.open(path)

    try:
        exif = image._getexif()
        if exif is not None:
            orientation_key = next(
                (k for k, v in ExifTags.TAGS.items() if v == "Orientation"), None
            )
            if orientation_key and orientation_key in exif:
                orientation = exif[orientation_key]
                if orientation == 3:
                    image = image.rotate(180, expand=True)
                elif orientation == 6:
                    image = image.rotate(270, expand=True)
                elif orientation == 8:
                    image = image.rotate(90, expand=True)
    except Exception as e:
        print(f"No EXIF rotation needed or error ignored: {e}")

    return np.array(image)

def classify_pose(yaw, pitch, roll):
    # Handle extreme tilt first
    if abs(roll) > 25:
        return "tilted"
    
    # Handle frontal
    if abs(yaw) < 15 and abs(pitch) < 15:
        return "frontal"
    
    # Handle left turns
    if yaw < -45:
        return "fullProfileLeft"
    elif yaw < -30:
        return "profileLeft"
    elif yaw < -15:
        return "slightLeft"
    
    # Handle right turns
    if yaw > 45:
        return "fullProfileRight"
    elif yaw > 30:
        return "profileRight"
    elif yaw > 15:
        return "slightRight"
    
    # If none of the above, call it angled
    return "angled"

def detect_and_embed_faces_for_photo(photo_path: str, partner: str, photo_id: str, override_image: np.ndarray = None):

    try:
        # image = load_image_with_exif_rotation(photo_path)
        image = override_image if override_image is not None else load_image_with_exif_rotation(photo_path)



        if image is None:
            raise Exception(f"Failed to load image {photo_path}")

        faces = face_app.get(image)
        # faces_dir = get_faces_dir(partner)
        # os.makedirs(faces_dir, exist_ok=True)

        face_embeddings = []
        faces_metadata = []
        for i, face in enumerate(faces):

            # Extract yaw, pitch, roll
            yaw = float(face.pose[0])
            pitch = float(face.pose[1])
            roll = float(face.pose[2])
            pose = classify_pose(yaw, pitch, roll)

            x1, y1, x2, y2 = face.bbox.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)

            try:
                # crop_img = image[y1:y2, x1:x2]
                # if crop_img is None or crop_img.size == 0:
                #     print(f"⚠️ Empty crop for face {i}")
                #     continue

                face_id = str(uuid.uuid4())
                # crop_path = os.path.join(faces_dir, f"{face_id}.jpg")
                # cv2.imwrite(crop_path, cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR))

                embedding = face.normed_embedding
                if embedding is None or len(embedding) == 0:
                    print(f"⚠️ No embedding for face {i}")
                    continue
                if np.linalg.norm(embedding) != 1:
                    embedding = embedding / np.linalg.norm(embedding)

                # embedding = embedding / np.linalg.norm(embedding)

                face_embeddings.append(embedding)

                faces_metadata.append({
                    "faceId": face_id,
                    "photoId": photo_id,
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    # "cropPath": crop_path,
                    # "sourceImagePath": photo_path,
                    "gender": int(face.gender),
                    "age": int(face.age),
                    "det_score": float(face.det_score),
                    "yaw": yaw,
                    "pitch": pitch,
                    "roll": roll,
                    "pose": pose,
                    # "vectorId": global_vector_id,  # ✅ GLOBAL
                    "embedding": [float(x) for x in embedding.tolist()]  # ✅ convert each to native float
                })

                # global_vector_id += 1  # ✅ Increment after each face

            except Exception as e:
                print(f"Error during cropping for face {i}: {str(e)}")
                continue


        # embeddings_dir = get_embeddings_dir(partner)
        # insightface_npy_path = os.path.join(embeddings_dir, f"{partner}_insightface.npy")
        # os.makedirs(embeddings_dir, exist_ok=True)

        # if os.path.exists(insightface_npy_path):
        #     existing = np.load(insightface_npy_path)
        #     combined = np.vstack([existing, np.array(face_embeddings, dtype="float32")])
        # else:
        #     combined = np.array(face_embeddings, dtype="float32")

        # for i, face in enumerate(faces_metadata):
        #     face["embedding"] = face_embeddings[i]
        #     face["partner"] = partner

        # # Save to SQLite
        # save_faces_to_db(faces_metadata)
       
        # metadata_path = get_metadata_path(partner)
        # metadata = load_jsonl(metadata_path)
        # for entry in metadata:
        #     if entry["id"] == photo_id:
        #         entry["faceCount"] = len(faces)
        #         break
        # # save_jsonl(metadata, metadata_path)

        return faces_metadata # ✅ return updated

    except Exception as e:
        log_error(
            photoId=photo_id,
            faceId=None,
            step="insightface",
            errorMessage=str(e),
            traceback=traceback.format_exc(),
            modelType="insightface",
            path=photo_path,
            partner=partner
        )
        return global_vector_id  # ✅ in case of error still return

def process_faces_folder_fast(partner: str):
    photos_dir = get_photos_dir(partner)
    metadata_path = get_metadata_path(partner)
    embeddings_dir = get_embeddings_dir(partner)
    faces_dir = get_faces_dir(partner)

    os.makedirs(embeddings_dir, exist_ok=True)
    os.makedirs(faces_dir, exist_ok=True)

    photo_files = [f for f in os.listdir(photos_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    metadata = load_jsonl(metadata_path)
    id_to_metadata = {item["id"]: item for item in metadata}

    all_embeddings = []
    all_faces = []
    processed = skipped = 0
    vector_id = 0

    for idx, file in enumerate(photo_files):
        photo_path = os.path.join(photos_dir, file)
        photo_id = os.path.splitext(file)[0]
        meta = id_to_metadata.get(photo_id)

        if not meta or meta.get("faceCount") is not None:
            skipped += 1
            continue

        image = load_image_with_exif_rotation(photo_path)
        faces = face_app.get(image)

        for i, face in enumerate(faces):
            embedding = face.normed_embedding
            if embedding is None:
                continue
            embedding = embedding / np.linalg.norm(embedding)

            yaw, pitch, roll = float(face.pose[0]), float(face.pose[1]), float(face.pose[2])
            pose = classify_pose(yaw, pitch, roll)

            face_id = str(uuid.uuid4())
            all_embeddings.append(embedding)
            all_faces.append({
                "faceId": face_id,
                "photoId": photo_id,
                "bbox": face.bbox.astype(int).tolist(),
                "sourceImagePath": photo_path,
                "gender": int(face.gender),
                "age": int(face.age),
                "detScore": float(face.det_score),
                "yaw": yaw,
                "pitch": pitch,
                "roll": roll,
                "pose": pose,
                "vectorId": vector_id
            })
            vector_id += 1

        # Update in-memory metadata
        meta["faceCount"] = len(faces)
        processed += 1

        if (processed + skipped) % 25 == 0:
            print(f"📸 Processed: {processed}, Skipped: {skipped}")

    # Save all at once
    embeddings_path = os.path.join(embeddings_dir, f"{partner}_insightface.npy")
    np.save(embeddings_path, np.array(all_embeddings, dtype="float32"))

    faces_metadata_path = os.path.join(faces_dir, f"{partner}_faces_metadata.jsonl")
    save_jsonl(all_faces, faces_metadata_path)

    save_jsonl(list(id_to_metadata.values()), metadata_path)

    print(f"\n✅ Done. Processed: {processed}, Skipped: {skipped}, Total: {len(photo_files)}")
    return {
        "processed": processed,
        "skipped": skipped,
        "total": len(photo_files)
    }

def process_faces_folder(partner: str):

    photos_dir = get_photos_dir(partner)
    metadata_path = get_metadata_path(partner)

    if not os.path.exists(photos_dir):
        raise Exception(f"No photos found at {photos_dir}")

    if not os.path.exists(metadata_path):
        raise Exception(f"Metadata file {metadata_path} not found. Run clip embedding first to generate metadata.")

    photo_files = [f for f in os.listdir(photos_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    metadata = load_jsonl(metadata_path)
    id_to_metadata = {item['id']: item for item in metadata}

    processed_count = 0
    skipped_count = 0
    global_vector_id = 0  # ✅ START GLOBAL COUNTER

    for idx, file in enumerate(photo_files):
        full_path = os.path.join(photos_dir, file)
        photo_id = os.path.splitext(file)[0]

        meta = id_to_metadata.get(photo_id)
        if not meta:
            continue
        if meta.get("faceCount") is not None:
            skipped_count += 1
            continue

        print("calling detect")
        global_vector_id = detect_and_embed_faces_for_photo(full_path, partner, photo_id, global_vector_id)
        processed_count += 1

        if (processed_count + skipped_count) % 10 == 0:
            print(f"Processed {processed_count}, skipped {skipped_count} out of {len(photo_files)} photos...")
        
    return {
        "processed": processed_count,
        "skipped": skipped_count,
        "total": len(photo_files)
    }

def detect_and_embed_faces_from_array(
    image: np.ndarray,
    partner: str,
    photo_id: str,
):
    photo_path = f"memory/{photo_id}"

    return detect_and_embed_faces_for_photo(photo_path, partner, photo_id, override_image=image)
