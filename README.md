# Local RAG Document Assistant

Ask questions about your own PDFs. Answers are generated entirely on-device with **Phi-3.5-mini** and grounded in retrieved chunks. Sources (file, page, and preview snippet) print under every reply.

No API keys required. After the initial model download, it runs fully offline on an NVIDIA GPU.

![Demo](https://github.com/user-attachments/assets/c6be59f9-d6c1-4c0c-8cff-2e11c973277b)

---

## How It Works

1. Upload one or more text-based PDFs in the Gradio UI.
2. Pages are split into ~800-character chunks (with 150-character overlap).
3. Chunks are embedded using `all-MiniLM-L6-v2` and stored locally in Chroma.
4. A user question retrieves the top-3 most relevant chunks.
5. Phi-3.5-mini answers using **only** that context.
6. Filename, page number, and snippet sources are displayed directly under the answer.

*This is a working retrieval pipeline, not a wrapped external chatbot API.*

---

## Technical Limitations

* **Hardware Requirement:** Requires an NVIDIA GPU and a **CUDA build** of PyTorch (CPU wheels will crash on `device_map="cuda"`).
* **PDF Types:** Supports text-based PDFs only. Scanned or image-only PDFs will index as empty and fail to retrieve.
* **File Sizing:** Very large files (100MB+) are a poor first test; start with a few small documents.
* **Environment:** Designed for a single local user via Gradio UI (no authentication or Docker containers yet).
* **Dependencies:** Uses `langchain-community` (currently in maintenance, but loaders remain functional).

---

## Quick Start

### 1. Environment Setup

```bash
python -m venv .venv

# Windows activation:
.venv\Scripts\activate

# Install GPU PyTorch (use cu130 for modern RTX GPUs / newer drivers)
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu130](https://download.pytorch.org/whl/cu130)
pip install accelerate langchain langchain-community langchain-huggingface chromadb pypdf sentence-transformers gradio transformers
