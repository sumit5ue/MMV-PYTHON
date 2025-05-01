# config.py

import os

BASE_DIR = "/Users/sumit/Documents/ai_analysis"

def get_partner_dir(partner: str) -> str:
    return os.path.join(BASE_DIR, partner)

def get_photos_dir(partner: str) -> str:
    return os.path.join(get_partner_dir(partner), "photos")

def get_faces_dir(partner: str) -> str:
    return os.path.join(get_partner_dir(partner), "faces")

def get_embeddings_dir(partner: str) -> str:
    return os.path.join(get_partner_dir(partner), "embeddings")

def get_metadata_path(partner: str) -> str:
    return os.path.join(get_partner_dir(partner), "metadata.jsonl")

def get_faces_metadata_path(partner: str) -> str:
    return os.path.join(get_partner_dir(partner), "faces", f"{partner}_faces_metadata.jsonl")

def get_errors_path(partner: str) -> str:
    return os.path.join(get_partner_dir(partner), "errors.jsonl")
