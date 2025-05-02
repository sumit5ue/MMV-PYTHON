from pathlib import Path
import sqlite3
import os

DB_PATH = Path("/Users/sumit/Documents/ai_analysis/data.db")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS cluster_versions (
        versionId TEXT PRIMARY KEY,
        partner TEXT,
        model TEXT,
        method TEXT,
        createdAt TEXT,
        summary TEXT,
        isLatest INTEGER
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS face_clusters (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        versionId TEXT,
        clusterId TEXT,
        faceId TEXT,
        confidence REAL,
        isNoise INTEGER,
        FOREIGN KEY (versionId) REFERENCES cluster_versions(versionId) ON DELETE CASCADE,
        FOREIGN KEY (faceId) REFERENCES face_embeddings(faceId) ON DELETE CASCADE
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS cluster_reps (
        versionId TEXT,
        clusterId TEXT,
        model TEXT,
        rep TEXT,
        PRIMARY KEY (versionId, clusterId, model),
        FOREIGN KEY (rep) REFERENCES face_embeddings(faceId) ON DELETE CASCADE,
        FOREIGN KEY (versionId) REFERENCES cluster_versions(versionId) ON DELETE CASCADE
    )
    """)

    conn.commit()
    conn.close()

get_conn()
