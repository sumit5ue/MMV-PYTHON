import json
import os
from typing import List
from pydantic import ValidationError
import requests
from pathlib import Path

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

def transform_to_photo_model(json_data: dict, partner: str) -> List[Photo]:
    """Transform JSON data to Photo model instances."""
    photos = []
    
    for photo_data in json_data.get('photos', []):
        shared_with = [kid['id'] if isinstance(kid, dict) and 'id' in kid else kid for kid in photo_data.get('kids', [])]
        if photo_data.get('rosterId'):
            shared_with.append(photo_data['rosterId'])
        
        try:
            photo = Photo(
                id=photo_data['id'],
                partner=partner,
                url = (
                        photo_data.get('video_file_url')
                        if photo_data.get('is_video')
                        else photo_data.get('main_url').replace('/main/', '/original/')
                        ),
                sharedWith=shared_with,
                rosterId=photo_data.get('rosterId'),
                caption=photo_data.get('caption'),
                isVideo=photo_data.get('is_video')
            )
            photos.append(photo)
        except ValidationError as e:
            print(f"Validation error for photo {photo_data['id']}: {str(e)}")
    
    return photos

def send_bulk_photos(photos: List[Photo], endpoint: str, headers: dict, batch_size: int = 500) -> bool:
    """Send bulk photos to the specified endpoint in batches."""
    success = True
    total_photos = len(photos)
    
    for i in range(0, total_photos, batch_size):
        batch = photos[i:i + batch_size]
        try:
            payload = [photo.dict(exclude_none=True) for photo in batch]
            
            response = requests.post(
                endpoint,
                json=payload,
                headers=headers
            )
            
            if response.status_code == 200:
                print(f"Successfully uploaded batch {i//batch_size + 1} ({len(batch)} photos)")
            else:
                print(f"Failed to upload batch {i//batch_size + 1}. Status: {response.status_code}, Response: {response.text}")
                success = False
                
        except Exception as e:
            print(f"Error sending batch {i//batch_size + 1}: {str(e)}")
            success = False
    
    return success

def main(folder_path: str, endpoint: str, partner: str, api_key: str, batch_size: int = 500):
    """Main function to process JSON files and upload photos."""
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    
    json_files = read_json_files(folder_path)
    
    all_photos = []
    for json_data in json_files:
        photos = transform_to_photo_model(json_data, partner)
        all_photos.extend(photos)
    
    if all_photos:
        success = send_bulk_photos(all_photos, endpoint, headers, batch_size)
        if success:
            print(f"Completed processing {len(all_photos)} photos")
        else:
            print("Failed to complete photo upload")
    else:
        print("No photos found to upload")

if __name__ == "__main__":
    FOLDER_PATH = "/Users/sumit/Documents/ai_analysis/67c5079afb7ebb148255e275/app_files/photos"
    ENDPOINT = "http://127.0.0.1:8000/db/photos/bulk"
    PARTNER = "67c5079afb7ebb148255e275"
    API_KEY = "your-api-key-here"  # Replace with actual API key
    BATCH_SIZE = 500
    
    main(FOLDER_PATH, ENDPOINT, PARTNER, API_KEY, BATCH_SIZE)