from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import sqlite3
import uuid
import datetime
import httpx
import numpy as np
import traceback
from services.clustering_service import run_hdbscan, log_error

DB_PATH = "/Users/sumit/Documents/ai_analysis/data.db"

router = APIRouter()

# --------- Models ---------

class ClusteredFace(BaseModel):
    faceId: str
    clusterId: str
    confidence: Optional[float] = None
    isNoise: Optional[bool] = False

class ClusterRequest(BaseModel):
    partner: str
    model: str
    method: str
    clusteredFaces: List[ClusteredFace]
    reps: List[str]

class ClusterRunRequest(BaseModel):
    partner: str
    method: Optional[str] = "hdbscan"
    distance: Optional[str] = "euclidean"
    minClusterSize: Optional[int] = 2


class ClusterSummaryRequest(BaseModel):
    partner: str
    model: str


# --------- Helpers ---------

def get_conn():
    return sqlite3.connect(DB_PATH)

# --------- Cluster Run ---------

def build_clusters(labels, metadata, membership_strength, model, partner):
    from collections import defaultdict
    from operator import itemgetter

    clusters = defaultdict(list)
    results = []

    for idx, label in enumerate(labels):
        meta = metadata[idx]
        cluster_id = str(label) if label != -1 else f"noise-{idx}"
        confidence = float(membership_strength[idx]) if membership_strength is not None else 0.0
        results.append({
            "faceId": meta["faceId"],
            "clusterId": cluster_id,
            "confidence": confidence,
            "isNoise": label == -1
        })
        if label != -1:
            clusters[cluster_id].append((confidence, meta["faceId"]))

    reps = [max(items, key=itemgetter(0))[1] for cluster_id, items in clusters.items()]

    return {
        "clusteredFaces": results,
        "reps": reps
    }

@router.post("/cluster/insightface")
async def cluster_faces_insightface(req: ClusterRunRequest):
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("""
            SELECT faceId, embedding FROM face_embeddings
            WHERE partner = ? AND embedding IS NOT NULL
        """, (req.partner,))
        rows = cur.fetchall()
        conn.close()

        if not rows:
            raise HTTPException(status_code=404, detail="No embeddings found for this partner")

        face_ids = [row[0] for row in rows]
        embeddings = np.array([np.frombuffer(row[1], dtype=np.float32) for row in rows])

        if req.distance == "cosine":
            norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.clip(norm, 1e-10, None)

        labels, membership_strength = run_hdbscan(
            embeddings,
            min_cluster_size=req.minClusterSize or 2,
            metric=req.distance or "euclidean"
        )

        metadata = [{"faceId": fid} for fid in face_ids]
        result = build_clusters(labels, metadata, membership_strength, "insightface", req.partner)

        # Construct payload as a ClusterRequest model
        clustered_faces = [ClusteredFace(**face) for face in result["clusteredFaces"]]
        payload = ClusterRequest(
            partner=req.partner,
            model="insightface",
            method=req.method,
            clusteredFaces=clustered_faces,
            reps=result["reps"]
        )

        # Call save_cluster_results directly
        result = await save_cluster_results(payload)
        return result

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
        raise HTTPException(status_code=500, detail=str(e))
# --------- Save cluster results ---------

@router.post("/db/cluster/faces")
async def save_cluster_results(body: ClusterRequest):
    try:
        version_id = str(uuid.uuid4())
        now = datetime.datetime.utcnow().isoformat()

        conn = get_conn()
        cur = conn.cursor()

        cur.execute("""
            INSERT INTO cluster_versions (versionId, partner, model, method, createdAt, isLatest)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (version_id, body.partner, body.model, body.method, now, 1))

        cur.execute("""
            UPDATE cluster_versions SET isLatest = 0
            WHERE partner = ? AND model = ? AND versionId != ?
        """, (body.partner, body.model, version_id))

        cluster_items = [
            (version_id, f.clusterId, f.faceId, f.confidence or 0.0, int(f.isNoise))
            for f in body.clusteredFaces
        ]
        cur.executemany("""
            INSERT INTO face_clusters (versionId, clusterId, faceId, confidence, isNoise)
            VALUES (?, ?, ?, ?, ?)
        """, cluster_items)

        reps_map = {}
        for face in body.clusteredFaces:
            if face.clusterId not in reps_map:
                reps_map[face.clusterId] = None
        for rep in body.reps:
            for cluster_id in reps_map:
                if reps_map[cluster_id] is None:
                    reps_map[cluster_id] = rep
                    break

        rep_items = [
            (version_id, cluster_id, body.model, rep_id)
            for cluster_id, rep_id in reps_map.items()
        ]
        cur.executemany("""
            INSERT INTO cluster_reps (versionId, clusterId, model, rep)
            VALUES (?, ?, ?, ?)
        """, rep_items)

        conn.commit()
        return {"status": "ok", "versionId": version_id, "clusters": len(reps_map)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

@router.post("/db/cluster/summary")
async def get_cluster_summary_post(req: ClusterSummaryRequest):
    try:
        conn = get_conn()
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        cur.execute("""
            SELECT versionId FROM cluster_versions
            WHERE partner = ? AND model = ? AND isLatest = 1
        """, (req.partner, req.model))
        row = cur.fetchone()

        if not row:
            raise HTTPException(status_code=404, detail="No clustering version found")

        version_id = row["versionId"]

        cur.execute("""
            SELECT fc.clusterId, fc.faceId, fc.confidence, fc.isNoise,
                   fe.photoId, fe.bb_x1, fe.bb_y1, fe.bb_x2, fe.bb_y2, fe.detScore,
                   fe.partner,
                   ph.path as photoPath
            FROM face_clusters fc
            JOIN face_embeddings fe ON fc.faceId = fe.faceId
            JOIN photos ph ON fe.photoId = ph.id
            WHERE fc.versionId = ?
        """, (version_id,))
        face_rows = cur.fetchall()

        from collections import defaultdict
        cluster_map = defaultdict(list)
        noise_faces = []

        for row in face_rows:
            face_data = {
                "faceId": row["faceId"],
                "photoId": row["photoId"],
                "confidence": row["confidence"],
                "isNoise": bool(row["isNoise"]),
                "bbox": [row["bb_x1"], row["bb_y1"], row["bb_x2"], row["bb_y2"]],
                "detScore": row["detScore"],
                "path": row["photoPath"]
            }
            if face_data["isNoise"]:
                noise_faces.append(face_data)
            else:
                cluster_map[row["clusterId"]].append(face_data)

        cur.execute("""
            SELECT clusterId, rep FROM cluster_reps
            WHERE versionId = ? AND model = ?
        """, (version_id, req.model))
        rep_map = {row[0]: row[1] for row in cur.fetchall()}

        summary = []

        # Add noise cluster first (clusterId = 'noise')
        if noise_faces:
            rep = max(noise_faces, key=lambda f: f["detScore"])['faceId']
            summary.append({
                "clusterId": "noise",
                "numFaces": len(noise_faces),
                "avgConfidence": sum(f["confidence"] for f in noise_faces) / len(noise_faces),
                "rep": rep,
                "faces": noise_faces
            })

        # Then add real clusters, sorted by size descending
        sorted_clusters = sorted(cluster_map.items(), key=lambda kv: -len(kv[1]))
        for clusterId, faces in sorted_clusters:
            rep = max(faces, key=lambda f: f["detScore"])["faceId"]
            summary.append({
                "clusterId": clusterId,
                "numFaces": len(faces),
                "avgConfidence": sum(f["confidence"] for f in faces) / len(faces),
                "rep": rep,
                "faces": faces
            })

        return {
            "partner": req.partner,
            "model": req.model,
            "versionId": version_id,
            "summary": {
                "totalClusters": len(summary),
                "clusters": summary
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()
