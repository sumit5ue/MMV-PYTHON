import json
import os
from typing import List
from pydantic import ValidationError
import requests
from pathlib import Path
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Use sys.path to resolve routes.photos import
import sys
sys.path.append(str(Path(__file__).parent.parent))  # Add /Users/sumit/MMV-PYTHON to sys.path
try:
    from routes.photos import SaliencyBox, Photo
except ImportError as e:
    print(f"Import error: {e}. Ensure routes/photos.py exists in /Users/sumit/MMV-PYTHON/routes/")
    raise

def read_json_files(folder_path: str) -> List[dict]:
    """Read all JSON files from the specified folder."""
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

def transform_to_photo_model(json_data: dict, partner: str) -> tuple[List[Photo], List[dict]]:
    """Transform JSON data to Photo model instances, filtering null/invalid sharedWith IDs and collecting errors."""
    photos = []
    errors = []
    
    for photo_data in json_data.get('photos', []):
        shared_with = []
        invalid_shared_with = []
        
        # Process kid IDs
        for kid in photo_data.get('kids', []):
            kid_id = kid['id'] if isinstance(kid, dict) and 'id' in kid else kid
            if kid_id and isinstance(kid_id, str) and kid_id.strip():
                shared_with.append(kid_id)
            else:
                invalid_shared_with.append(kid_id)
        
        # Process rosterId
        roster_id = photo_data.get('rosterId')
        if roster_id and isinstance(roster_id, str) and roster_id.strip():
            shared_with.append(roster_id)
        elif roster_id:
            invalid_shared_with.append(roster_id)
        
        # Log and collect invalid sharedWith IDs
        if invalid_shared_with:
            print(f"Invalid sharedWith IDs for photo {photo_data['id']}: {invalid_shared_with}")
            errors.append({"photoId": photo_data['id'], "invalid_ids": invalid_shared_with})
        
        if not shared_with:
            print(f"Warning: No valid sharedWith IDs for photo {photo_data['id']}")
        
        try:
            photo = Photo(
                id=photo_data['id'],
                partner=partner,
                url=(
                    photo_data.get('video_file_url')
                    if photo_data.get('is_video')
                    else photo_data.get('main_url').replace('/main/', '/original/')
                ),
                sharedWith=shared_with,
                rosterId=photo_data.get('rosterId'),
                caption=photo_data.get('caption'),
                isVideo=photo_data.get('is_video'),
                createdAt=photo_data.get("created_at")
            )
            photos.append(photo)
        except ValidationError as e:
            print(f"Validation error for photo {photo_data['id']}: {str(e)}")
            errors.append({"photoId": photo_data['id'], "error": str(e)})
    
    return photos, errors

def send_bulk_photos(photos: List[Photo], endpoint: str, headers: dict, batch_size: int = 500) -> tuple[bool, int, int, List[dict]]:
    """Send bulk photos to the endpoint, logging photos and sharedWith counts, collecting errors."""
    success = True
    uploaded_photos = 0
    uploaded_shared_with = 0
    errors = []
    
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    session.mount('http://', HTTPAdapter(max_retries=retries))
    
    for i in range(0, len(photos), batch_size):
        batch = photos[i:i + batch_size]
        batch_shared_with = sum(len(photo.sharedWith) for photo in batch)
        try:
            payload = [photo.dict(exclude_none=True) for photo in batch]
            print(f"Sending batch {i//batch_size + 1}: {len(batch)} photos, {batch_shared_with} sharedWith entries")
            
            response = session.post(
                endpoint,
                json=payload,
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                print(f"Successfully uploaded batch {i//batch_size + 1}: {len(batch)} photos, {batch_shared_with} sharedWith entries")
                uploaded_photos += len(batch)
                uploaded_shared_with += batch_shared_with
                # Check for endpoint errors in response
                response_data = response.json()
                if "errors" in response_data:
                    errors.extend(response_data["errors"])
            else:
                error_msg = f"Status: {response.status_code}, Response: {response.text}"
                print(f"Failed to upload batch {i//batch_size + 1}. {error_msg}")
                errors.append({"batch_number": i//batch_size + 1, "error": error_msg})
                success = False
                
        except Exception as e:
            error_msg = str(e)
            print(f"Error sending batch {i//batch_size + 1}: {error_msg}")
            errors.append({"batch_number": i//batch_size + 1, "error": error_msg})
            success = False
    
    return success, uploaded_photos, uploaded_shared_with, errors

def save_errors_to_json(errors: dict, filename: str = "errors.json"):
    """Save errors to a JSON file."""
    try:
        with open(filename, 'w') as f:
            json.dump(errors, f, indent=2)
        print(f"Errors saved to {filename}")
    except Exception as e:
        print(f"Failed to save errors to {filename}: {str(e)}")

def main(folder_path: str, endpoint: str, partner: str, api_key: str, batch_size: int = 500):
    """Main function to process JSON files, upload photos, and consolidate errors."""
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    
    # Initialize error collection
    all_errors = {
        "invalid_shared_with": [],
        "batch_upload_errors": [],
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
    
    # Merge sharedWith lists for duplicate photoIds
    photo_dict = {}
    for photo in all_photos:
        if photo.id in photo_dict:
            existing_photo = photo_dict[photo.id]
            merged_shared_with = list(set(existing_photo.sharedWith + photo.sharedWith))
            photo_dict[photo.id] = Photo(
                id=photo.id,
                partner=photo.partner,
                url=photo.url,
                sharedWith=merged_shared_with,
                rosterId=photo.rosterId,
                caption=photo.caption,
                isVideo=photo.isVideo,
                created_at=photo.created_at
            )
        else:
            photo_dict[photo.id] = photo
    
    # Convert back to list of unique photos
    deduped_photos = list(photo_dict.values())
    
    # Print unique photo count and sharedWith details
    print("deduped_photos: unique count is ---", len(deduped_photos))
    total_shared_with = 0
    print("sharedWith counts per photoId:")
    for photo in deduped_photos:
        shared_with_count = len(photo.sharedWith)
        total_shared_with += shared_with_count
        print(f"  photoId: {photo.id}, sharedWith count: {shared_with_count}")
        if shared_with_count > 50:
            print(f"  WARNING: Large sharedWith count for photoId: {photo.id}, count: {shared_with_count}")
    print("Total sharedWith entries after merging ---", total_shared_with)
    
    if deduped_photos:
        success, uploaded_photos, uploaded_shared_with, batch_errors = send_bulk_photos(deduped_photos, endpoint, headers, batch_size)
        all_errors["batch_upload_errors"].extend(batch_errors)
        print(f"Uploaded {uploaded_photos} photos with {uploaded_shared_with} sharedWith entries")
        if success:
            print(f"Completed processing {len(deduped_photos)} photos")
        else:
            print("Failed to complete photo upload")
    else:
        print("No photos found to upload")
    
    # Save errors to JSON
    save_errors_to_json(all_errors, "errors.json")
    return all_errors

if __name__ == "__main__":
    FOLDER_PATH = "/Users/sumit/Documents/ai_analysis/67c5079afb7ebb148255e275/app_files/photos"
    ENDPOINT = "http://127.0.0.1:8000/db/photos/bulk"
    PARTNER = "67c5079afb7ebb148255e275"
    API_KEY = "your-api-key-here"  # Replace with actual API key
    BATCH_SIZE = 500
    
    main(FOLDER_PATH, ENDPOINT, PARTNER, API_KEY, BATCH_SIZE)