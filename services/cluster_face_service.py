import os
import uuid
from PIL import Image
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import Column, String, Integer, Float, DateTime, UUID, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func
import uuid
import traceback
from collections import defaultdict

from sklearn.cluster import HDBSCAN
import hdbscan
# from models.cluster_face_rep import  ClusterFaceRep
from models.photo import ClusteringRun, ClusterFaceRepFaceMapping

from models.photo import Face,ClusterFaceRep
# from models.cluster_face_rep import ClusterFaceRep
from utils.get_latest_cluster_run import get_latest_cluster_run

# Constant for minimum detection score
MIN_DET_SCORE = 0.6
# Constant for minimum bounding box dimensions
MIN_BBOX_DIMENSION = 80


def get_rep_faces(session: Session, partner: str):
    """
    Get the representative faces for the latest clustering run of the given partner.
    """
    try:
        # Step 1: Get the latest clustering run for the partner
        latest_run = get_latest_cluster_run(session, partner)

        # Step 2: Query the ClusterFaceRep table for the representative faces for the latest run
        rep_faces = session.execute(
            select(ClusterFaceRep.run_id, ClusterFaceRep.cluster_id, ClusterFaceRep.rep_face_id,ClusterFaceRep.cluster_label)
            .filter(ClusterFaceRep.run_id == latest_run.run_id)
        ).all()

        if not rep_faces:
            raise ValueError(f"No representative faces found for the latest run for partner: {partner}")

        # Format the result into a list of dictionaries
        return [{"run_id": row.run_id, "cluster_id": row.cluster_id, "rep_face_id": row.rep_face_id,"label":row.cluster_label} for row in rep_faces]
    
    except Exception as e:
        raise ValueError(f"Error fetching representative faces for partner {partner}: {str(e)}")


def get_run_cluster_faces(session, run_id, label):
    try:
        # Step 1: Query the ClusterFaceRepFaceMapping table joined with Face table
        faces = session.execute(
            select(
                ClusterFaceRepFaceMapping.face_id,
                ClusterFaceRepFaceMapping.label,
                ClusterFaceRepFaceMapping.aws_face_id,
                ClusterFaceRepFaceMapping.confidence,
                ClusterFaceRepFaceMapping.run_id,
                Face.photo_id,  # Include photo_id from Face table
                Face.bbox_width,
                Face.bbox_height,
                Face.bbox_x,
                Face.bbox_y
            )
            .join(Face, ClusterFaceRepFaceMapping.face_id == Face.face_id)  # Join on face_id
            .filter(
                ClusterFaceRepFaceMapping.label == label,
                ClusterFaceRepFaceMapping.run_id == run_id  # Corrected filter to use run_id parameter
            )
        ).all()

        if not faces:
            raise ValueError(f"No faces found for run_id: {run_id} and label: {label}")

        # Step 2: Format the result into a list of dictionaries
        return [
            {
                "run_id": row.run_id,
                "label": row.label,
                "face_id": row.face_id,
                "photo_id": row.photo_id,
                "confidence": row.confidence,
                "aws_face_id": row.aws_face_id,
                "bbox_x":row.bbox_x,
                "bbox_y":row.bbox_y,
                "bbox_width":row.bbox_width,
                "bbox_height":row.bbox_height,
                
            }
            for row in faces
        ]

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        raise  # Re-raise the exception for upstream handling
    
def get_faces_for_partner(session: Session, partner: str):
    """
    Retrieve faces for a given partner, filtering by:
    - det_score >= 0.6
    - bbox width and height >= 40 pixels
    - Non-null embeddings (for HDBSCAN compatibility)
    
    Args:
        session (Session): SQLAlchemy session
        partner (str): Partner identifier
    
    Returns:
        list: List of Face objects with valid embeddings
    """
    # Query faces with filters
    faces = session.execute(
        select(Face)
        .filter(
            Face.partner == partner,
            Face.det_score >= MIN_DET_SCORE,
            # Ensure bbox width (width) >= 40
            Face.bbox_width  >= MIN_BBOX_DIMENSION,
            # Ensure bbox height (height) >= 40
             Face.bbox_height >= MIN_BBOX_DIMENSION,
            Face.embedding.isnot(None)  # Ensure embedding is not null for HDBSCAN
        )
    ).scalars().all()

    return faces

def process_faces_with_hdbscan(faces):
    # Extract embeddings from faces
    faces_subset = faces[:1000]
        
    # Extract embeddings from the faces
    # embeddings = np.array([face.embedding for face in faces_subset])
    embeddings = np.array([face.embedding for face in faces])
    norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
    # embeddings = embeddings / np.clip(norm, 1e-10, None)
        
    # Log the number of faces and the shape of the embeddings array
    # print(f"Number of faces: {len(faces_subset)}")
    # print(f"Embeddings shape: {embeddings.shape}")

    # embeddings = np.array([face.embedding for face in faces])
    print(f"Embeddings shape: {embeddings.shape}")

    # Run HDBSCAN clustering
    # clusterer = hdbscan.HDBSCAN(
    #     # min_cluster_size=min_cluster_size,
    #     # min_samples=5, for insightface
    #     min_samples=1,
    #     metric="euclidean",
    #     cluster_selection_method='eom',
    #     prediction_data=True,
    #     # cluster_selection_epsilon=0,
    #     min_cluster_size=2,
    #     cluster_selection_epsilon=0.0 # for insightface
    # )
    clusterer = hdbscan.HDBSCAN(
        min_samples=1,
        min_cluster_size=2,
        metric="euclidean",
        cluster_selection_method='leaf',
        prediction_data=True,
        cluster_selection_epsilon=0
    )
    # Fit and predict cluster labels
    labels = clusterer.fit_predict(embeddings)
    label_to_photos = defaultdict(list)
    for photo, label in zip(faces, labels):
        label_to_photos[label].append(photo.photo_id)

        # Print the grouped results
    print("\nCluster assignments (label: [photo_id]):")
    for label in sorted(label_to_photos.keys()):  # Sort for consistent output
        photo_ids = label_to_photos[label]
        print(f"Label {label}: {photo_ids} (Count: {len(photo_ids)})")
    
    # Get the cluster labels
    # labels = clustering.labels_
    return labels, clusterer

def crop_and_save_face(face, partner: str):
    """
    Crops the face from the image using its bounding box and saves it to the desktop.
    """
    image_path = f"/Users/sumit/Documents/ai_analysis/{partner}/photos/{face.photo_id}.jpg"  # Replace with actual logic to load the image.
    image = Image.open(image_path)
    
    # Get the bounding box coordinates for cropping
    left = face.bbox_x  # Get the left position of the bounding box
    top = face.bbox_y   # Get the top position of the bounding box (added consistency with face.box)
    right = left + face.bbox_width  # Calculate the right position by adding width from the face.box
    bottom = top + face.bbox_height  # Calculate the bottom position by adding height from the face.box
    # right = left + face.bb_width
    # bottom = top + face.bb_height
    
    # Crop the image
    cropped_image = image.crop((left, top, right, bottom))
    
    # Create the folder for saving if it doesn't exist
    save_dir = f"/Users/sumit/Documents/ai_analysis/{partner}/face_reps"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Save the cropped face
    save_path = os.path.join(save_dir, f"{face.face_id}.jpg")
    cropped_image.save(save_path)

# def save_cluster_faces(session: Session, faces, labels, clustering, partner: str):
#     cluster_faces = []
    
#     # Generate a unique UUID for the cluster version (only once per cluster)
#     cluster_ver = str(uuid.uuid4())  # Generate UUID for the cluster version
#     cluster_faces_in_groups = {}
#     noise_faces = []
#     # For each cluster, including noise (label -1)
#     for cluster_id in set(labels):
#         # cluster_faces_in_group = [face for i, face in enumerate(faces) if labels[i] == cluster_id]
#         cluster_faces_in_group = [face for i, face in enumerate(faces) if labels[i] == cluster_id]

#         # If the cluster_id is -1, group it as noise
#         if cluster_id == -1:
#             noise_faces.extend(cluster_faces_in_group)
#         else:
#             cluster_faces_in_groups[cluster_id] = cluster_faces_in_group
#         # If there are faces in the cluster, select the one with the highest det_score
#         if cluster_faces_in_group:
#             rep_face = max(cluster_faces_in_group, key=lambda f: f.det_score)
            
#             # Crop and save the representative face image
#             crop_and_save_face(rep_face, partner)
            
#             # Store each face in a separate record
#             for face in cluster_faces_in_group:
#                 cluster_face = FaceCluster(
#                     partner=partner,
#                     cluster_ver=cluster_ver,  # Use the same UUID for all faces in this cluster
#                     cluster_id=str(cluster_id),
#                     rep_id=rep_face.id,  # Representative face ID for the entire cluster
#                     cluster_item_id=face.id,  # ID of the face in this cluster
#                     confidence=float(clustering.probabilities_[faces.index(face)]),  # Convert np.float64 to Python float
#                     member_id=face.member_id,
#                     roster_id=face.roster_id,
#                     aws_face_id=face.aws_face_id,
#                     data=face.data,  # Additional data, if needed
#                 )
#                 cluster_faces.append(cluster_face)
    
#     # Bulk insert into the database
#     session.bulk_save_objects(cluster_faces)
#     session.commit()

def assign_representative_face_and_crop(faces, labels,partner):
    # Step 1: Group faces by cluster label
    clustered_faces = defaultdict(list)
    for i, label in enumerate(labels):
        clustered_faces[label].append(faces[i])

    # Step 2: For each cluster (unique label), find the face with the highest det_score
    representative_faces = {}
    for label, faces_in_cluster in clustered_faces.items():
        # For noise label (e.g., label = -1), we treat it similarly
        highest_score_face = max(faces_in_cluster, key=lambda f: f.det_score)

        # Store the face_id as the representative face for this cluster (or noise)
        representative_faces[label] = highest_score_face.face_id
        
        # Crop and save the representative face
        crop_and_save_face(highest_score_face,partner)
        
    return representative_faces


def save_cluster_rep_faces(db: Session, faces, labels, clustering, partner, clustering_run: ClusteringRun):
    cluster_face_reps = []  # List to store ClusterFaceRep instances for returning later
    cluster_label_to_cluster_id = {} 
    try:
        # Step 1: Identify representative faces (and crop them)
        rep_faces = assign_representative_face_and_crop(faces, labels, partner)
        
        # Step 2: Save the cluster results to the database (loop over rep_faces dictionary)
        for cluster_label, rep_face_id in rep_faces.items():
            print("Cluster label is", cluster_label)
            cluster_id = uuid.uuid4()  # Create a new UUID for the cluster
           
            # Create the ClusterFace record for each representative face
            cluster_face = ClusterFaceRep(
                cluster_id=cluster_id,
                run_id=clustering_run.run_id,
                partner=partner,
                cluster_label = str(cluster_label) if isinstance(cluster_label, np.int64) else cluster_label,
                rep_face_id=rep_face_id,  # The representative face ID for this cluster
                metadata={"label": cluster_label},
                aws_face_id=None,  # Set other face details as needed
                aws_external_image_id=None,
            )
            cluster_label_to_cluster_id[cluster_label] = cluster_id
            cluster_face_reps.append(cluster_face)  # Add to the list of cluster face reps
            db.add(cluster_face)  # Add to the session for commit

        # Commit the changes to the database
        db.commit()

        # Return the list of created ClusterFaceRep instances
        return cluster_label_to_cluster_id

    except Exception as e:
        db.rollback()  # Rollback the transaction in case of error
        print(f"Error while saving cluster representative faces: {str(e)}")
        raise Exception(f"Error while saving cluster representative faces for partner {partner}: {str(e)}")

# def save_cluster_faces(db: Session, faces, labels, clustering, partner, clustering_run: ClusteringRun):
#     """Saves the cluster face records."""
#     for i, face in enumerate(faces):
#         cluster_label = labels[i]
#         centroid = clustering.cluster_centers_[cluster_label] if cluster_label != -1 else None
#         print("centroid is---",centroid)
#         # Create and save ClusterFace record
#         cluster_face = ClusterFace(
#             run_id=clustering_run.run_id,
#             partner=partner,
#             rep_face_id=face.face_id,
#             centroid=centroid,
#             metadata={"label": cluster_label},
#             aws_face_id=face.aws_face_id,
#             aws_external_image_id=face.aws_external_image_id,
#         )

#         db.add(cluster_face)
#         db.commit()
#         db.refresh(cluster_face)  # Ensure we get the cluster_id after commit


def cluster_faces_for_partner(db: Session, partner: str):
    # Step 1: Get faces for the given partner
    try:
        # Step 1: Get faces for the given partner
        faces = get_faces_for_partner(db, partner)
        print("face count is", len(faces))
        if not faces:
            raise ValueError("No faces found for partner")

        # Step 2: Process faces with HDBSCAN
        labels, clustering = process_faces_with_hdbscan(faces)
        

        # Step 3: Create ClusteringRun (this should be within try block)
        clustering_run = create_clustering_run(db, partner)

        # Step 4: Save the cluster faces in the ClusterFace table
        cluster_label_to_cluster_id = save_cluster_rep_faces(db, faces, labels, clustering, partner, clustering_run)

        # Step 5: Save the cluster-face mappings
        save_cluster_face_mappings(db, faces, labels,clustering,clustering_run, partner,cluster_label_to_cluster_id)

        return {"message": f"Clustering for partner {partner} completed and saved."}

    # except SQLAlchemyError as e:
    #     db.rollback()
    #     print(f"Error: {str(e)}")
    #     raise Exception(f"Error while saving clustering data for {partner}. Please check the logs.")
    # except ValueError as e:
    #     print(f"Error: {str(e)}")
    #     raise e  # Re-raise to handle error at a higher level, if necessary
    except Exception as e:
        db.rollback()  # Rollback in case of error
        print(f"Error in cluster faces: {str(e)}")
        traceback.print_exc()
        return {"message": f"Error creating cluster : {str(e)}"}



def get_unique_rep_faces(db: Session, partner: str, cluster_ver: str):
    # Query the FaceCluster table to get rep_id, cluster_id, and confidence for the given partner and cluster_ver
    unique_rep_faces = db.execute(
        select(FaceCluster.rep_id, FaceCluster.cluster_id, FaceCluster.confidence)
        .filter(FaceCluster.partner == partner, FaceCluster.cluster_ver == cluster_ver)
    ).all()

    if not unique_rep_faces:
        return []

    # Use a dictionary to ensure unique rep_id (the dictionary keys will be rep_id)
    unique_rep_dict = {}
    for rep_id, cluster_id, confidence in unique_rep_faces:
        # Only store the first occurrence of each unique rep_id
        if rep_id not in unique_rep_dict:
            unique_rep_dict[rep_id] = {
                "rep_id": rep_id,
                "cluster_id": cluster_id,
                "confidence": confidence
            }

    # Prepare the response data
    response_data = []
    for rep_id, cluster_info in unique_rep_dict.items():
        face = db.execute(select(Face).filter(Face.id == rep_id)).scalars().first()
        if face:
            response_data.append({
                "rep_id": rep_id,
                "face_id": face.id,  # Assuming the column in the Face model is called 'id'
                "clusterId": cluster_info["cluster_id"],
                "det_score": face.det_score,  # Assuming the column for detection score is 'det_score'
                "confidence": cluster_info["confidence"]
            })

    return response_data


def get_faces_for_rep_id(db: Session, rep_id: str):
    # Query the FaceCluster table for the given rep_id
    face_clusters = db.execute(
        select(FaceCluster).filter(FaceCluster.rep_id == rep_id)
    ).scalars().all()

    if not face_clusters:
        return []

    # For each FaceCluster item, get the face details (face_id, det_score)
    response_data = []
    for face_cluster in face_clusters:
        face = db.execute(select(Face).filter(Face.id == face_cluster.cluster_item_id)).scalars().first()
        if face:
            response_data.append({
                "face_id": face.face_id,  # Assuming the column in Face model is 'face_id'
                "det_score": face.det_score,  # Assuming the column in Face model is 'det_score'
                "confidence": face_cluster.confidence,  # Confidence comes from FaceCluster
                "cluster_id": face_cluster.cluster_id,
                "rep_id": face_cluster.rep_id,
            })

    return response_data


def save_cluster_face_mappings(db: Session, faces, labels, clustering, clustering_run, partner: str,cluster_label_to_cluster_id):
    print("in func", len(faces))
    try:
        # Loop through each face and assign a cluster label based on the clustering
        for i, face in enumerate(faces):
            cluster_label = labels[i]  # Get the cluster label for the current face from the labels list
            
            # Ensure cluster_label and confidence are the correct types
            cluster_label = int(cluster_label)  # Convert to native Python int
            confidence = clustering.probabilities_[i] if hasattr(clustering, 'probabilities_') and len(clustering.probabilities_) > i else None
            confidence = float(confidence) if confidence is not None else None  # Convert to native Python float if not None

            # Get the cluster_id from the map
            cluster_id = cluster_label_to_cluster_id.get(cluster_label)

            if cluster_id is None:
                print(f"Cluster ID not found for label {cluster_label}")
                continue  # If no cluster ID found for this label, skip


            try:
                # Create a ClusterFaceRepFaceMapping for each face
                cluster_face_mapping = ClusterFaceRepFaceMapping(
                    face_id=face.face_id,
                    label=cluster_label,
                    run_id=clustering_run.run_id,
                    partner=partner,
                    confidence=confidence,
                    cluster_id=cluster_id,
                    # aws_face_id=face.aws_face_id,  # Uncomment if needed
                    # data=face.data  # Uncomment if needed
                )


                # Add the mapping to the session
                db.add(cluster_face_mapping)

            except AttributeError as ae:
                print(f"AttributeError in ClusterFaceRepFaceMapping: {str(ae)}")
                raise
            except TypeError as te:
                print(f"TypeError in ClusterFaceRepFaceMapping: {str(te)}")
                raise
            except IndexError as ie:
                print(f"IndexError in ClusterFaceRepFaceMapping: {str(ie)}")
                raise
            except Exception as e:
                print(f"Unexpected error in ClusterFaceRepFaceMapping: {str(e)}")
                raise

        # Commit all changes to the database after processing all faces
        db.commit()
        print("Cluster face mappings successfully saved.")

        return {"message": "Cluster face mappings created successfully"}

    except Exception as e:
        db.rollback()  # Rollback in case of error
        print(f"Error creating cluster face mappings: {str(e)}")
        return {"message": f"Error creating cluster face mappings: {str(e)}"}

def create_clustering_run(db: Session, partner: str) -> ClusteringRun:
    """Creates a new ClusteringRun record."""
    clustering_run = ClusteringRun(
        partner=partner,
        cluster_item_type="face",  # Can adjust based on the item type
        algorithm="HDBSCAN",
        parameters={"min_cluster_size": 5, "min_samples": 3},  # Example params
    )
    db.add(clustering_run)
    db.commit()
    db.refresh(clustering_run)  # Ensure we get the run_id after the commit
    return clustering_run


    """Saves the cluster-face mappings."""
    for face in faces:
        cluster_face_mapping = ClusterFaceRepFaceMapping(
            face_id=face.face_id,
            cluster_id=face.cluster_id,
            partner=partner,
            confidence=face.confidence,  # Assuming confidence is available            
        )

        db.add(cluster_face_mapping)
        db.commit()