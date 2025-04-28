import json

def compare_aesthetic_scores(file1_path, file2_path, output_path):
    # Load file1
    with open(file1_path, 'r') as f1:
        file1_data = {}
        for line in f1:
            obj = json.loads(line)
            file1_data[obj["fileName"]] = obj["aestheticScore"]

    # Load file2
    with open(file2_path, 'r') as f2:
        file2_data = {}
        for line in f2:
            obj = json.loads(line)
            file2_data[obj["fileName"]] = obj["aestheticScore"]

    # Compare based on fileName
    common_filenames = set(file1_data.keys()) & set(file2_data.keys())

    with open(output_path, 'w') as out:
        for filename in sorted(common_filenames):
            output_obj = {
                "fileName": filename,
                "score1": file1_data[filename],
                "score2": file2_data[filename]
            }
            out.write(json.dumps(output_obj) + "\n")

    print(f"✅ Comparison written to {output_path}")

# Example usage
compare_aesthetic_scores(
    "/Users/sumit/Documents/ai_analysis/67c507b9fb7ebb148255e4af/metadata.jsonl",
   "/Users/sumit/Documents/ai_analysis/67c507b9fb7ebb148255e4af/metadata2.jsonl",
   "/Users/sumit/Documents/ai_analysis/67c507b9fb7ebb148255e4af/metadata_compare.jsonl",
)
