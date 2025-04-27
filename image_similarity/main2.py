from fastapi import FastAPI
from image_similarity.routers.face_detect import router as face_detect_router
from image_similarity.routers.face_index import router as face_index_router
from image_similarity.routers.face_search import router as face_search_router
from image_similarity.routers.face_pipeline import router as face_pipeline_router

app = FastAPI()

# Register routers
app.include_router(face_detect_router)
app.include_router(face_index_router)
app.include_router(face_search_router)
app.include_router(face_pipeline_router)
