echo "# Image Similarity Project

This project uses FastAPI, DINO, FAISS, and MongoDB to perform image similarity search.

## Setup

1. Create and activate a virtual environment:
   \`\`\`bash
   python3 -m venv venv
   source venv/bin/activate
   \`\`\`
2. Install dependencies:
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`
3. Update the MongoDB URI in \`image_similarity/models.py\`.
4. Run the server:
   \`\`\`bash
   uvicorn image_similarity.main:app --host 0.0.0.0 --port 8000
   \`\`\`

## Endpoints

- **POST /embed**: Embed an image and store its features.
- **POST /similar**: Find similar images based on a query image.
  " > README.md
# mmv-python
