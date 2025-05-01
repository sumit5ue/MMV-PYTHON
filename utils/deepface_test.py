import os
import json
from deepface import DeepFace

# Folder of face crops
folder_path = os.path.expanduser("~/Desktop/exp")

results = []

# Loop over all .jpg files
for file_name in os.listdir(folder_path):
    # if not file_name.lower().endswith(".jpg"):
    #     continue

    image_path = os.path.join(folder_path, file_name)
    print(f"Analyzing: {file_name}")

    try:
        result = DeepFace.analyze(
            img_path=image_path,
            actions=["emotion"],
            enforce_detection=False
        )

        emotions = result[0]["emotion"]
        dominant_emotion = max(emotions.items(), key=lambda x: float(x[1]))

        results.append({
            "file": file_name,
            "dominant_emotion": dominant_emotion[0],
            "confidence": round(float(dominant_emotion[1]), 2)
        })

    except Exception as e:
        print(f"Error analyzing {file_name}: {e}")
        results.append({
            "file": file_name,
            "error": str(e)
        })

# Save to JSON
output_path = os.path.join(folder_path, "emotion_results.json")
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"✅ Saved emotion results to {output_path}")
