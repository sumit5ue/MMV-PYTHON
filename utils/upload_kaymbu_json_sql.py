import json
import os
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
from pydantic import ValidationError
import sys

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

# ---- Import your models ----
from schemas.photo import PhotoCreate
from models.photo import Photo  # SQLAlchemy model
from db.session import SessionLocal  # DB session factory


def read_json_files(folder_path: str) -> List[dict]:
    json_data = []
    folder = Path(folder_path)
    if not folder.exists():
        print(f"Folder {folder_path} does not exist.")
        return json_data
    for file_path in folder.glob("*.json"):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                json_data.append(data)
        except Exception as e:
            print(f"Error reading {file_path}: {str(e)}")
    return json_data


def transform_to_photo_model(json_data: dict, partner: str) -> tuple[List[PhotoCreate], List[dict]]:
    photos = []
    errors = []

    for photo_data in json_data.get('photos', []):
        shared_with = []
        invalid_shared_with = []

        # for kid in photo_data.get('students', []):
        #     kid_id = kid['id'] if isinstance(kid, dict) and 'id' in kid else kid
        #     if kid_id and isinstance(kid_id, str) and kid_id.strip():
        #         shared_with.append(kid_id)
        #     else:
        #         invalid_shared_with.append(kid_id)

        # app_roster_id = photo_data.get('rosterId')
        # if app_roster_id and isinstance(app_roster_id, str) and app_roster_id.strip():
        #     shared_with.append(app_roster_id)
        # elif app_roster_id:
        #     invalid_shared_with.append(app_roster_id)

        # if invalid_shared_with:
        #     print(f"Invalid shared_with IDs for photo {photo_data['id']}: {invalid_shared_with}")
        #     errors.append({"photoId": photo_data['id'], "invalid_ids": invalid_shared_with})

        try:
            photo = PhotoCreate(
                photo_id=photo_data['_id'],
                partner=partner,
                # url=(
                #     photo_data.get('video_file_url')
                #     if photo_data.get('is_video')
                #     else photo_data.get('pictureThumb').replace('/_thumb/', '//')
                # ),
                # shared_with=shared_with,
                # app_roster_id=photo_data.get('rosterId'),
                url = f"https://d2k9f6tk478nyp.cloudfront.net{photo_data.get('pictureThumb').replace('_thumb', '')}",

                caption=photo_data.get('annotation'),
                is_video=photo_data.get('video'),
                photo_creation_date=photo_data.get("dateCaptured"),
            )
            photos.append(photo)
        except ValidationError as e:
            print(f"Validation error for photo {photo_data['id']}: {str(e)}")
            errors.append({"photoId": photo_data['id'], "error": str(e)})

    return photos, errors


def insert_photos_to_db(photo_models: List[PhotoCreate], batch_size=1000):
    session = SessionLocal()
    try:
        for i in range(0, len(photo_models), batch_size):
            batch = photo_models[i:i + batch_size]
            db_rows = [Photo(**photo.dict(exclude_none=True)) for photo in batch]
            session.bulk_save_objects(db_rows)
            session.commit()
            print(f"✅ Inserted batch {i//batch_size + 1}: {len(batch)} photos")
    except Exception as e:
        session.rollback()
        print(f"❌ Error inserting batch: {str(e)}")
    finally:
        session.close()


def save_errors_to_json(errors: dict, filename: str = "errors.json"):
    try:
        with open(filename, 'w') as f:
            json.dump(errors, f, indent=2)
        print(f"Errors saved to {filename}")
    except Exception as e:
        print(f"Failed to save errors to {filename}: {str(e)}")


def main(folder_path: str, partner: str):
    all_errors = {
        "invalid_shared_with": [],
        "validation_errors": []
    }

    json_files = read_json_files(folder_path)

    all_photos = []
    for json_data in json_files:
        photos, transform_errors = transform_to_photo_model(json_data, partner)
        all_photos.extend(photos)
        all_errors["invalid_shared_with"].extend([e for e in transform_errors if "invalid_ids" in e])
        all_errors["validation_errors"].extend([e for e in transform_errors if "error" in e])

    print("all_photos: total count is ---", len(all_photos))

    # Deduplicate photo IDs and merge shared_with
    photo_dict = {}
    for photo in all_photos:
        if photo.photo_id in photo_dict:
            existing = photo_dict[photo.photo_id]
            merged = list(set(existing.shared_with + photo.shared_with))
            photo_dict[photo.photo_id] = PhotoCreate(
                photo_id=photo.photo_id,
                partner=photo.partner,
                url=photo.url,
                shared_with=merged,
                app_roster_id=photo.app_roster_id,
                caption=photo.caption,
                is_video=photo.is_video,
                photo_creation_date=photo.photo_creation_date,
            )
        else:
            photo_dict[photo.photo_id] = photo

    deduped_photos = list(photo_dict.values())
    print("deduped_photos: unique count is ---", len(deduped_photos))
    print("photo is ",deduped_photos[0])

    # Insert to DB
    insert_photos_to_db(deduped_photos, batch_size=1000)

    save_errors_to_json(all_errors, "errors.json")
    print("✅ Done!")


if __name__ == "__main__":
    FOLDER_PATH = "/Users/sumit/Documents/ai_analysis/67ec2a60d4d64df971004210/app_files/photos"
    PARTNER = "67ec2a60d4d64df971004210"
    main(FOLDER_PATH, PARTNER)
