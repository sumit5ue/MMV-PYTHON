import boto3
from sqlalchemy.orm import Session
from PIL import Image
from io import BytesIO
from models.photo import Face
from utils.file_utils import get_photo_path
from config import get_photos_dir
from db.session import SessionLocal
from sqlalchemy import or_


MIN_BBOX_DIMENSION = 60

# Initialize AWS Rekognition client using a specific profile
def get_rekognition_client(profile_name='default'):
    session = boto3.Session(profile_name=profile_name)  # Create a session using the specific profile
    rekognition_client = session.client('rekognition', region_name='us-west-2')  # Set your desired region
    return rekognition_client

# Initialize Rekognition client
rekognition_client = get_rekognition_client('default')  # You can specify the profile name here


async def process_faces_for_partner(partner: str):
    # Query faces where partner matches and aws_face_id is null
    db = SessionLocal() 

    faces = db.query(Face).filter(
        Face.partner == partner,
        Face.aws_face_id == None,  # Correct way to check for non-NULL
        Face.is_aws_error == False,  # Check that 'is_aws_error' is not True
        or_(
            Face.bbox_width >= MIN_BBOX_DIMENSION, 
            Face.bbox_height >= MIN_BBOX_DIMENSION
        )
    ).limit(1000).all()
    
    if not faces:
        return {"message": "No faces to process."}
    
    processed_faces = []
    
    # Iterate through each face to process it
    for face in faces:
        try:
            # Construct full image path directly from the face's photo_id
            image_path = get_photo_path(partner, face.photo_id)  # Get the full image path for the face
            image = Image.open(image_path)

            # Crop the face based on the bounding box
            cropped_face = image.crop((
                face.bbox_x, 
                face.bbox_y, 
                face.bbox_x + face.bbox_width, 
                face.bbox_y + face.bbox_height
            ))

            # Convert the cropped face to bytes for AWS Rekognition
            with BytesIO() as img_byte_arr:
                cropped_face.save(img_byte_arr, format='JPEG')
                img_byte_arr.seek(0)

                # Call AWS Rekognition to search for faces in the collection
                response = rekognition_client.search_faces_by_image(
                    CollectionId=partner,  # The collection where the face is indexed
                    Image={'Bytes': img_byte_arr.read()},  # The cropped face image in byte format
                    MaxFaces=1,  # The maximum number of faces to return
                    FaceMatchThreshold=90  # The minimum similarity threshold to consider a match (adjust as needed)
                )

            # Check if Rekognition returned faces
            if response.get('FaceMatches') and len(response['FaceMatches']) > 0:
                matched_face = response['FaceMatches'][0]  # Get the most confident match (if any)
                print("matched face is",matched_face)
                aws_face_id = matched_face['Face']['FaceId']
                # external_image_id = matched_face['Face']['ExternalImageId'] or None
                similarity = matched_face['Similarity']  # Similarity score for the match

                # Prepare the data to update the face in the database
                face.aws_face_id = aws_face_id
                # face.external_image_id = external_image_id
                face.similarity = similarity

                # Clear any previous errors
                face.aws_error = {}  
                face.is_aws_error = False

                # Commit changes to the database
                db.commit()

                # Append processed face data to the result list
                processed_faces.append({
                    'face_id': face.face_id,
                    'aws_face_id': aws_face_id,
                    # 'external_image_id': external_image_id,
                    'similarity': similarity,
                    'aws_error': face.aws_error
                })

            else:
                # No match found, record error details
                face.aws_error = {'message': 'No face match found'}
                face.is_aws_error = True
                db.commit()

                # Log the error and continue to the next face
                processed_faces.append({
                    'face_id': face.face_id,
                    'aws_error': face.aws_error
                })
                continue  # Skip updating this face and move to the next one

        except Exception as e:
            # Handle any errors during the process
            face.aws_error = {'message': str(e)}
            face.is_aws_error = True
            db.commit()

            # Append error details to the result list
            processed_faces.append({
                'face_id': face.face_id,
                'aws_error': face.aws_error
            })


    return { "processed_faces": processed_faces}
