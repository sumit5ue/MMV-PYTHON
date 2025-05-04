import psycopg2
import jsonlines
from psycopg2.extras import Json

# PostgreSQL connection settings
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "vector_db"
DB_USER = "postgres"
DB_PASSWORD = "sumit123"

# Connect to the PostgreSQL database
def get_db_connection():
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        print("Connected to database: vector_db")
        return conn
    except Exception as e:
        print(f"Failed to connect to database: {e}")
        raise

# Function to read and process JSONL file
def process_jsonl(file_path):
    with jsonlines.open(file_path) as reader:
        for line in reader:
            process_record(line)

# Function to process each record and update the database
def process_record(record):
    photo_id = record.get('id')
    aesthetic_score = record.get('aestheticScore')
    sal = record.get('attentionSaliency')

    # Initialize query components
    set_clauses = []
    params = []
    updates = []

    # Process aesthetic_score if present
    if aesthetic_score is not None:
        set_clauses.append("aesthetic_score = %s")
        params.append(aesthetic_score)
        updates.append("aesthetic_score")

    # Process saliency if present
    attention_saliency_json = None
    if sal and isinstance(sal, list) and len(sal) > 0 and isinstance(sal[0], list):
        saliency = {
            "x": sal[0][0],
            "y": sal[0][1],
            "width": sal[0][2],
            "height": sal[0][3]
        }
        print("saliency", saliency)
        attention_saliency_json = Json(saliency)
        print("attention_saliency_json", attention_saliency_json)
        set_clauses.append("saliency = %s")
        params.append(attention_saliency_json)
        updates.append("saliency")
    else:
        print(f"No valid saliency data for photo {photo_id}")

    # Skip update if no fields to update
    if not set_clauses:
        print(f"Skipping update for photo {photo_id}: no valid aesthetic_score or saliency data")
        return

    # Construct the query
    query = f"""
        UPDATE public.photos
        SET {', '.join(set_clauses)}
        WHERE photo_id = %s
    """
    params.append(photo_id)

    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(query, params)
        conn.commit()
        print(f"Updated photo {photo_id} with {', '.join(updates)}.")
    except Exception as e:
        print(f"Error updating photo {photo_id}: {e}")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

# Main function to run the process
if __name__ == "__main__":
    jsonl_file_path = '/Users/sumit/Documents/ai_analysis/67c5079afb7ebb148255e275/photos/metadata.jsonl'
    process_jsonl(jsonl_file_path)