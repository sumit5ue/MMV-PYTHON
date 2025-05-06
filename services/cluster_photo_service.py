import os
import uuid
from PIL import Image
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import select,case
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import Column, String, Integer, Float, DateTime, UUID, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func
import uuid
import traceback


from collections import defaultdict

from sklearn.cluster import HDBSCAN
import hdbscan
from models.photo import ClusterPhotoRep,ClusteringRun,ClusterClip

from models.photo import Photo,ClusterPhotoRepPhotoMapping, ClusterClip
from models.clip import Clip
from utils.get_latest_cluster_run import get_latest_cluster_run
# from models.cluster_photo_rep import ClusterPhotoRep


def get_run_cluster_photos(session: Session, partner: str):
    try:
        # Step 1: Retrieve the latest clustering run for the given partner
        latest_run = get_latest_cluster_run(session, partner)  # Assuming this function returns the latest run
        
        # Step 2: Query the ClusterClip table for entries matching the run_id and partner using ORM
        cluster_clips = session.query(ClusterClip) \
            .filter(ClusterClip.run_id == latest_run.run_id, ClusterClip.partner == partner) \
            .all()

        if not cluster_clips:
            raise ValueError(f"No cluster clips found for partner {partner}")

        # Step 3: Group the results by label
        grouped_by_label = defaultdict(list)

        for clip in cluster_clips:
            grouped_by_label[clip.label].append({
                "run_id": clip.run_id,
                "label": clip.label,
                "photo_id": clip.photo_id,
                "confidence": clip.confidence,
                "category": clip.category,
                "chapter": clip.chapter,
                "data": clip.data
            })

        # Return the grouped result
        return dict(grouped_by_label)  # Convert defaultdict to a regular dict for better readability
    
    except Exception as e:
        print(f"Error {e}")
        traceback.print_exc()
        return None  # Optional: Return None or an empty list depending on how you want to handle errors
    
def get_rep_photos(session: Session, partner: str):
    """
    Get the representative faces for the latest clustering run of the given partner.
    """
    try:
        # Step 1: Get the latest clustering run for the partner
        latest_run = get_latest_cluster_run(session, partner)

        # Step 2: Query the ClusterFaceRep table for the representative faces for the latest run
        rep_photos = session.execute(
            select(ClusterPhotoRep.run_id, ClusterPhotoRep.cluster_id, ClusterPhotoRep.rep_photo_id,ClusterPhotoRep.cluster_label)
            .filter(ClusterPhotoRep.run_id == latest_run.run_id)
        ).all()
        print("rep_photos",rep_photos)
        if not rep_photos:
            raise ValueError(f"No representative faces found for the latest run for partner: {partner}")

        # Format the result into a list of dictionaries
        return [{"run_id": row.run_id, "cluster_id": row.cluster_id, "rep_photo_id": row.rep_photo_id,"cluster_label":row.cluster_label} for row in rep_photos]
    
    except Exception as e:
        print(f"Error {e}")
        traceback.print_exc()


# def get_run_cluster_photos(session, run_id,label):
#     try:
#         # Step 2: Query the ClusterFaceRep table for the representative faces for the latest run
#         photos = session.execute(
#             select(ClusterPhotoRepPhotoMapping.photo_id, ClusterPhotoRepPhotoMapping.label,ClusterPhotoRepPhotoMapping.chapter_id,ClusterPhotoRepPhotoMapping.confidence,ClusterPhotoRepPhotoMapping.run_id)
#             .filter(ClusterPhotoRepPhotoMapping.label == label,ClusterPhotoRepPhotoMapping.run_id == ClusterPhotoRepPhotoMapping.run_id)
#         ).all()
#         print(photos)
#         if not photos:
#             raise ValueError(f"No  photos found for the latest run: {run_id} and label {label}")

#         # Format the result into a list of dictionaries
#         return [{"run_id": row.run_id, "label": row.label, "photo_id": row.photo_id,"confidence":row.confidence} for row in photos]
    
#     except Exception as e:
#         print(f"Error {e}")
#         traceback.print_exc()


# def get_run_cluster_photos(session, run_id,label):
#     try:
#         # Step 2: Query the ClusterFaceRep table for the representative faces for the latest run
#         photos = session.execute(
#             select(ClusterPhotoRepPhotoMapping.photo_id, ClusterPhotoRepPhotoMapping.label,ClusterPhotoRepPhotoMapping.chapter_id,ClusterPhotoRepPhotoMapping.confidence,ClusterPhotoRepPhotoMapping.run_id)
#             .filter(ClusterPhotoRepPhotoMapping.label == label,ClusterPhotoRepPhotoMapping.run_id == ClusterPhotoRepPhotoMapping.run_id)
#         ).all()
#         print(photos)
#         if not photos:
#             raise ValueError(f"No  photos found for the latest run: {run_id} and label {label}")

#         # Format the result into a list of dictionaries
#         return [{"run_id": row.run_id, "label": row.label, "photo_id": row.photo_id,"confidence":row.confidence} for row in photos]
    
#     except Exception as e:
#         print(f"Error {e}")
#         traceback.print_exc()


def get_embeddings_for_partner(session: Session, partner: str):
    photos = session.query(Photo).filter(
        Photo.partner == partner, 
        Photo.is_clip_created == True
    ).all()

    print("photos are---",len(photos))
    # Create a list of photo_ids to preserve the order
    photo_ids = [photo.photo_id for photo in photos]

    # Query and order the results by the index of the photo_ids in the original list
    data = session.query(Clip).filter(Clip.photoId.in_(photo_ids)) \
        .order_by(case({photo_id: index for index, photo_id in enumerate(photo_ids)}, value=Clip.photoId)) \
        .all()
    # Query the Clip table to get embeddings for each photo_id
    # data = session.query(Clip).filter(Clip.photoId.in_([photo.photo_id for photo in photos])).all()

    return photos,data

def process_photos_with_hdbscan(data):
    # embeddings = np.array([face.embedding for face in faces_subset])
    embeddings = np.array([item.embedding for item in data])
    
    
    norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.clip(norm, 1e-10, None)
        
    # Log the number of faces and the shape of the embeddings array
    # print(f"Number of faces: {len(faces_subset)}")
    # print(f"Embeddings shape: {embeddings.shape}")

    # embeddings = np.array([face.embedding for face in faces])
    print(f"Embeddings shape: {embeddings.shape}")

    # Run HDBSCAN clustering
    clusterer = hdbscan.HDBSCAN(
        # min_cluster_size=min_cluster_size,
        # min_samples=5, for insightface
        min_samples=1,
        metric="euclidean",
        cluster_selection_method='eom',
        prediction_data=True,
        # cluster_selection_epsilon=0,
        min_cluster_size=2,
        cluster_selection_epsilon=0 # for insightface
    )
    # Fit and predict cluster labels
    labels = clusterer.fit_predict(embeddings)


    # Get the cluster labels
    # labels = clustering.labels_
    return labels, clusterer


def assign_representative_photo(photos, labels,partner):
    # Step 1: Group faces by cluster label
    clustered_photos = defaultdict(list)
    for i, label in enumerate(labels):
        clustered_photos[label].append(photos[i])

    # Step 2: For each cluster (unique label), find the face with the highest det_score
    representative_photos = {}
    for label, photos_in_cluster in clustered_photos.items():
        print("label, photos_in_cluster",label, photos_in_cluster)
        # For noise label (e.g., label = -1), we treat it similarly
        # highest_score_photo = max(photos_in_cluster, key=lambda f: f.det_score)
        highest_score_photo = photos[0]
        # Store the face_id as the representative face for this cluster (or noise)
        representative_photos[label] = photos_in_cluster[0].photo_id
        
       
    return representative_photos


def save_cluster_rep_photos(db: Session, photos, labels, clustering, partner, clustering_run: ClusteringRun):
    cluster_photo_reps = []  # List to store ClusterFaceRep instances for returning later
    
    try:
        # Step 1: Identify representative faces (and crop them)
        rep_photos = assign_representative_photo(photos, labels, partner)
        cluster_id_label = {}
        # Step 2: Save the cluster results to the database (loop over rep_faces dictionary)
        for cluster_label, rep_photo_id in rep_photos.items():
            print("Cluster label is", cluster_label)
            cluster_id = uuid.uuid4()  # Create a new UUID for the cluster
            cluster_id_label[cluster_label] = cluster_id
           
            # Create the ClusterFace record for each representative face
            cluster_photo = ClusterPhotoRep(
                cluster_id=cluster_id,
                run_id=clustering_run.run_id,
                partner=partner,
                cluster_label = str(cluster_label) if isinstance(cluster_label, np.int64) else cluster_label,
                rep_photo_id=rep_photo_id,  # The representative face ID for this cluster
                metadata={"label": cluster_label},
            )
            
            cluster_photo_reps.append(cluster_photo)  # Add to the list of cluster face reps
            db.add(cluster_photo)  # Add to the session for commit

        # Commit the changes to the database
        db.commit()

        # Return the list of created ClusterFaceRep instances
        return cluster_photo_reps,cluster_id_label

    except Exception as e:
        db.rollback()  # Rollback the transaction in case of error
        print(f"Error {e}")
        traceback.print_exc()


def cluster_photos_for_partner(db: Session, partner: str):
    # Step 1: Get faces for the given partner
    try:
        # Step 1: Get ordered embeddings for photos from Clip
        photos,embeddings = get_embeddings_for_partner(db, partner)
        print("embedding count is", len(embeddings))
        if not embeddings:
            raise ValueError("No faces found for partner")

        # Step 2: Process faces with HDBSCAN
        labels, clustering = process_photos_with_hdbscan(embeddings)
        
        # Group photos by cluster label and print photo_ids
        label_to_photos = defaultdict(list)
        for photo, label in zip(photos, labels):
            label_to_photos[label].append(photo.photo_id)

        # Print the grouped results
        print("\nCluster assignments (label: [photo_id]):")
        for label in sorted(label_to_photos.keys()):  # Sort for consistent output
            photo_ids = label_to_photos[label]
            print(f"Label {label}: {photo_ids} (Count: {len(photo_ids)})")

        # Step 3: Create ClusteringRun (this should be within try block)
        clustering_run = create_clustering_run(db, partner)
        print("labes are",labels)
        save_cluster_clip_data(db, label_to_photos, clustering_run)
        # Step 4: Save the cluster faces in the ClusterFace table
        # cluster_photo_reps,cluster_id_label=save_cluster_rep_photos(db, photos, labels, clustering, partner, clustering_run)

        # # Step 5: Save the cluster-face mappings
        # save_cluster_photo_mappings(db, photos, labels,clustering,clustering_run, partner,cluster_id_label)

        return {"message": f"Clustering for partner {partner} completed and saved."}

    except SQLAlchemyError as e:
        db.rollback()
        print(f"Error: {str(e)}")
        raise Exception(f"Error while saving clustering data for {partner}. Please check the logs.")
    except ValueError as e:
        print(f"Error: {str(e)}")
        raise e  # Re-raise to handle error at a higher level, if necessary

def save_cluster_clip_data(db: Session, label_to_photos: defaultdict, clustering_run: ClusteringRun):
    """Save the clustering results to the database."""
    
    # Iterate over each cluster label and its associated photos
    for label, photo_ids in label_to_photos.items():
        for photo_id in photo_ids:
            # Create a new ClusterClip for each photo in this cluster
            cluster_clip = ClusterClip(
                photo_id=photo_id,
                label=int(label),
                run_id=clustering_run.run_id,
                partner=clustering_run.partner,               
            )
            # Add it to the session and commit the changes
            db.add(cluster_clip)
    
    # Commit all the changes to the database
    db.commit()

def save_cluster_photo_mappings(db: Session, photos, labels, clustering, clustering_run, partner: str, cluster_id_label):
    print("in func", len(photos))
    try:
        # Loop through each photo and assign a cluster label based on the clustering
        for i, photo in enumerate(photos):
            cluster_label = labels[i]  # Get the cluster label for the current photo from the labels list
            
            # Ensure cluster_label and confidence are the correct types
            cluster_label = int(cluster_label)  # Convert to native Python int
            confidence = clustering.probabilities_[i] if hasattr(clustering, 'probabilities_') and len(clustering.probabilities_) > i else None
            confidence = float(confidence) if confidence is not None else None  # Convert to native Python float if not None

            try:
                # Create a ClusterPhotoRepPhotoMapping for each photo
                cluster_photo_mapping = ClusterPhotoRepPhotoMapping(
                    photo_id=photo.photo_id,
                    label=cluster_label,
                    run_id=clustering_run.run_id,
                    partner=partner,
                    confidence=confidence,
                    cluster_id=cluster_id_label.get(cluster_label)  # Safely get the cluster_id, avoid KeyError
                    # aws_face_id=face.aws_face_id,  # Uncomment if needed
                    # data=face.data  # Uncomment if needed
                )

                # print("cluster---", cluster_id_label.get(cluster_label), cluster_id=cluster_id_label.get(cluster_label))

                # Add the mapping to the session
                db.add(cluster_photo_mapping)

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

        # Commit all changes to the database after processing all photos
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
        cluster_item_type="photo",  # Can adjust based on the item type
        algorithm="HDBSCAN",
        parameters={"min_cluster_size": 5, "min_samples": 3},  # Example params
    )
    db.add(clustering_run)
    db.commit()
    db.refresh(clustering_run)  # Ensure we get the run_id after the commit
    return clustering_run


    # """Saves the cluster-face mappings."""
    # for face in faces:
    #     cluster_face_mapping = ClusterFaceRepFaceMapping(
    #         face_id=face.face_id,
    #         cluster_id=face.cluster_id,
    #         partner=partner,
    #         confidence=face.confidence,  # Assuming confidence is available            
    #     )

    #     db.add(cluster_face_mapping)
    #     db.commit()