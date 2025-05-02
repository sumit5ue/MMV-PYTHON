import sqlite3
import numpy as np
from pathlib import Path

DB_PATH = Path("/Users/sumit/Documents/ai_analysis/data.db")

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS face_embeddings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        faceId TEXT,
        photoId TEXT,
        partner TEXT,
        embedding BLOB,
        bb_x1 INTEGER,
        bb_y1 INTEGER,
        bb_x2 INTEGER,
        bb_y2 INTEGER,
        detScore REAL,
        gender INTEGER,
        age INTEGER,
        yaw REAL,
        pitch REAL,
        roll REAL,
        pose TEXT,
        clusterId TEXT,
        memberId TEXT,
        awsFaceId TEXT,
        FOREIGN KEY (photoId) REFERENCES photos(id) ON DELETE CASCADE
    )
    """)
    return conn

def save_faces_to_db(faces: list, batch_size: int = 20):
    if not faces:
        return

    conn = get_conn()
    cur = conn.cursor()

    for i in range(0, len(faces), batch_size):
        batch = faces[i:i + batch_size]
        cur.executemany("""
        INSERT INTO face_embeddings (
            faceId, photoId, partner, embedding,
            bb_x1, bb_y1, bb_x2, bb_y2,
            detScore, gender, age,
            yaw, pitch, roll, pose,
            clusterId, memberId, awsFaceId
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            (
                face["faceId"],
                face["photoId"],
                face["partner"],
                face["embedding"].astype("float32").tobytes(),
                face["bbox"][0], face["bbox"][1],
                face["bbox"][2], face["bbox"][3],
                face["detScore"],
                face["gender"],
                face["age"],
                face["yaw"],
                face["pitch"],
                face["roll"],
                face["pose"],
                face.get("clusterId"),
                face.get("memberId"),
                face.get("awsFaceId")
            )
            for face in batch
        ])
        conn.commit()

    conn.close()
if __name__ == "__main__":
    conn = get_conn()
    conn.close()
    print("✅ face_embeddings table checked/created.")