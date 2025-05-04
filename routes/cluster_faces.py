# routes/cluster_face_routes.py

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from services.cluster_face_service import cluster_faces_for_partner,get_rep_faces,get_faces_for_rep_id,get_run_cluster_faces
from db.session import get_db  # Import the get_db function for session management

router = APIRouter()

class PartnerRequest(BaseModel):
    partner: str

@router.post("/cluster_faces/")
def process_faces(request: PartnerRequest, db: Session = Depends(get_db)):  # Dependency injected here
    partner = request.partner
    try:
        # Call the service to handle the clustering and saving
        response = cluster_faces_for_partner(db, partner)
        return response
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/rep-faces/{partner}/")
async def get_rep_faces_route(partner: str, db: Session = Depends(get_db)):
    try:
        # Call the service to get the unique rep faces
        unique_rep_faces = get_rep_faces(db, partner)

        if not unique_rep_faces:
            raise HTTPException(status_code=404, detail="No unique representative faces found")

        return {"data": unique_rep_faces}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@router.get("/faces/{rep_id}")
async def get_faces_by_rep_id(rep_id: str, db: Session = Depends(get_db)):
    try:
        # Call the service to get the faces for the given rep_id
        faces = get_faces_for_rep_id(db, rep_id)

        if not faces:
            raise HTTPException(status_code=404, detail="No faces found for the given rep_id")

        return {"faces": faces}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@router.get("/cluster/faces/faces/{run_id}/{label}/")
async def get_run_cluster_faces_route(run_id: str, label: str,db: Session = Depends(get_db)):
    try:
        # Call the service to get the unique rep faces
        unique_rep_faces = get_run_cluster_faces(db, run_id,label)

        if not unique_rep_faces:
            raise HTTPException(status_code=404, detail="No unique representative faces found")

        return {"data": unique_rep_faces}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
