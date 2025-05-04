# MMV-PYTHON: Setup Guide for M1 Mac (Apple Silicon)

This guide will help you set up your environment cleanly for running FastAPI server, CLIP/DINO/InsightFace embeddings, and clustering.

---

## 1. Install Homebrew (if not already installed)

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

---

## 2. Install Miniforge (Lightweight Conda for M1)

```bash
brew install miniforge
```

---

## 3. Create Conda Environment

```bash
conda create -n mmv-python python=3.11
```

Activate the environment:

```bash
conda activate mmv-python
```

---

## 4. Install Packages

### 4.1 Install via Conda (precompiled)

```bash
conda install onnx onnxruntime opencv scikit-learn
```

### 4.2 Install via Pip (Python packages)

```bash
pip install insightface
pip install fastapi uvicorn torch torchvision Pillow
pip install git+https://github.com/openai/CLIP.git
```

✅ This separates heavy libraries (conda) and fast Python libraries (pip) cleanly.

---

## 5. Verify Installation

Check Python version:

```bash
python --version
```

It should show:

```plaintext
Python 3.11.x
```

List installed packages:

```bash
pip list
```

You should see:

- onnx
- onnxruntime
- insightface
- opencv
- scikit-learn
- torch
- torchvision
- fastapi
- uvicorn
- Pillow
- clip (installed from GitHub)

---

## 6. Start FastAPI Server

```bash
uvicorn main:app --reload
```

Access APIs like:

- `/clip/process-folder/`
- `/dino/process-folder/`
- `/faces/process-folder/`

✅ Server will hot-reload on changes.

---

## 7. Project Structure

```plaintext
MMV-PYTHON/
├── main.py
├── services/
│   ├── clip_service.py
│   ├── dino_service.py
│   ├── insightface_service.py
├── schemas/
│   ├── processing.py
├── utils/
│   ├── jsonl_utils.py
│   ├── error_utils.py
├── config.py
└── README.md
```

---

## 8. Troubleshooting

- Always run `conda activate mmv-python` before running uvicorn.
- Confirm your Python path inside the environment:
  ```bash
  which python
  ```
  It should point to `/miniforge3/envs/mmv-python/bin/python`.
- If you see `ModuleNotFoundError`, check `pip list` to ensure required packages are installed.
- Restart Terminal and `conda activate mmv-python` if environment seems inactive.
- Deactivate any old venvs:
  ```bash
  deactivate
  ```
- Ensure uvicorn is run **inside** the correct conda environment.

---

# Summary: Commands At A Glance

```bash
brew install miniforge
conda create -n mmv-python python=3.11
conda activate mmv-python
conda install onnx onnxruntime opencv scikit-learn
pip install insightface
pip install fastapi uvicorn torch torchvision Pillow
pip install git+https://github.com/openai/CLIP.git
uvicorn main:app --reload
```

✅ 10 minutes and you're ready to run a production-grade ML/AI pipeline on M1 Mac!

---

# ✅ You're now ready to:

- Embed photos (CLIP, DINO)
- Detect faces + expressions (InsightFace)
- Build clustering and search pipeline
- Serve everything via FastAPI

---

If you face any issues, remember to always:

- Check if conda environment is activated.
- Check installed packages.
- Restart terminal if needed after setting up Miniforge.

Ready to 🚀!

# psql -U postgres -d vector_db
