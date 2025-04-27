import os
from image_similarity.recognizer.face_recognizer import (
    index_known_faces,
    recognize_faces_in_image
)

TEST_IMAGE = "image_similarity/recognizer/test_inputs/test_photo.jpg"
KNOWN_FOLDER = "image_similarity/recognizer/known_faces"

def print_results(results):
    for idx, result in enumerate(results):
        print(f"\n🧠 Face {idx + 1} (bbox: {result['bbox']}):")
        for rank, match in enumerate(result["matches"], start=1):
            status = "✅ MATCH" if match["match"] else "❌"
            print(f"  {rank}. {match['name']} — similarity: {match['similarity']} {status}")

def main():
    print(f"\n📂 Indexing known faces from: {KNOWN_FOLDER}")
    known_faces = index_known_faces(KNOWN_FOLDER)

    if not known_faces:
        print("🚫 No known faces were indexed. Exiting.")
        return

    print(f"\n🖼️ Analyzing test image: {os.path.basename(TEST_IMAGE)}")
    results = recognize_faces_in_image(TEST_IMAGE, known_faces)

    if not results:
        print("🚫 No faces found in test image.")
    else:
        print_results(results)

if __name__ == "__main__":
    main()
