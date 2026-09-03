# Local RAG Document Assistant

**Brian Riendeau Jr.**

Ask questions about your own PDFs. Answers are generated on-device with **Phi-3.5-mini** and grounded in retrieved chunks. Sources (file, page, preview) print under every reply.

No API keys. After the first model download it runs offline on an NVIDIA GPU.

![Demo](https://github.com/user-attachments/assets/c6be59f9-d6c1-4c0c-8cff-2e11c973277b)

## What it does

1. Upload one or more text PDFs in the Gradio UI  
2. Pages are split into ~800-character chunks (150 overlap)  
3. Chunks are embedded with `all-MiniLM-L6-v2` and stored in Chroma  
4. A question retrieves the top-3 chunks  
5. Phi-3.5 answers using **only** that context  
6. Filename + page + snippet are shown under the answer  

This is a working retrieval pipeline, not a wrapped chatbot.

## Limits (read these)

- Needs an NVIDIA GPU and a **CUDA build** of PyTorch (CPU wheels will crash on `device_map="cuda"`)
- Text PDFs only. Scans / image-only files index as empty and fail
- Very large files (100MB+) are a poor first test; use a few small docs
- Single local user, Gradio UI, no auth, no Docker yet
- `langchain-community` is in maintenance; loaders still work

## Quick start

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate

# GPU PyTorch — use cu130 on RTX 50-series / new drivers
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
pip install accelerate langchain langchain-community langchain-huggingface chromadb pypdf sentence-transformers gradio transformers

python app.py
