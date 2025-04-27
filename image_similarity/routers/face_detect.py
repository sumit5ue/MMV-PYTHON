from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse,JSONResponse
from image_similarity.utils.face_utils import detect_faces
import os
import cv2

router = APIRouter(prefix="/detect-faces1", tags=["Face Detection"])

def cleanup_temp(path):
    if os.path.exists(path):
        os.remove(path)

@router.post("")
async def detect_faces_api(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    output_path = f"output_{file.filename}"
    
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    try:
        # Detect faces
        faces = detect_faces(temp_path)
        # print(faces)

        # Draw bounding boxes
        image = cv2.imread(temp_path)
        for face in faces:
            x1, y1, x2, y2 = face["bbox"]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Save output image
        cv2.imwrite(output_path, image)

        # Return image with faces drawn
        # return FileResponse(output_path, media_type="image/jpeg")

        # Optional: Return JSON instead (commented out)
        return JSONResponse(content={"faces": faces})

    finally:
        cleanup_temp(temp_path)
        # Optionally clean output after sending if needed
        # cleanup_temp(output_path)
