from pymongo import MongoClient
from bson import ObjectId

def initialize_mongo():
    # python -c "from pymongo import MongoClient; print(MongoClient('mongodb+srv://sumit5ue:SumitStar2024@cluster0.bpw9q.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0').list_database_names())"
    mongo_uri = "mongodb+srv://sumit5ue:SumitStar2024@cluster0.bpw9q.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    client = MongoClient(mongo_uri)
    db = client["image_db"]
    collection = db["images"]
    db.counters.update_one(
        {"_id": "faiss_id"},
        {"$setOnInsert": {"seq": -1}},  # Start at -1 so first increment is 0
        upsert=True
    )
    return client, collection

def get_next_id(collection):
    counter = collection.database["counters"].find_one_and_update(
        {"_id": "faiss_id"},
        {"$inc": {"seq": 1}},
        upsert=True,
        return_document=True
    )
    return counter["seq"]

def store_image_metadata(collection, metadata):
    result = collection.insert_one(metadata)
    return result.inserted_id

def find_image_metadata(collection, faiss_id):
    return collection.find_one({"faiss_id": faiss_id})