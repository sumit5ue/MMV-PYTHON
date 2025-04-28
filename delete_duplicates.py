import json

def deduplicate_faces(input_path, output_path):
    seen = set()
    deduped_records = []

    with open(input_path, "r") as f:
        for line in f:
            try:
                record = json.loads(line)
                # Define the deduplication key
                key = (record["photoId"], tuple(record["bbox"]))
                if key not in seen:
                    seen.add(key)
                    deduped_records.append(record)
            except json.JSONDecodeError:
                print(f"Skipping invalid line: {line.strip()}")

    with open(output_path, "w") as f:
        for record in deduped_records:
            f.write(json.dumps(record) + "\n")

    print(f"Deduplicated {len(deduped_records)} records saved to {output_path}")

# Example usage
deduplicate_faces(
    input_path="/Users/sumit/Documents/ai_analysis/67c507b9fb7ebb148255e4af/faces/67c507b9fb7ebb148255e4af_faces_metadata.jsonl",
    output_path="/Users/sumit/Documents/ai_analysis/67c507b9fb7ebb148255e4af/faces/67c507b9fb7ebb148255e4af_faces_metadata_clean.jsonl"
)
