Developed By Brian Riendeau Jr. &copy;

# Local RAG Chatbot - A AI Document Assistant Application

DEMO (Upload any PDF documents you want the LLM to answer questions about):
<img width="3738" height="2055" alt="Rag1" src="https://github.com/user-attachments/assets/c6be59f9-d6c1-4c0c-8cff-2e11c973277b" />


A fully local Retrieval-Augmented Generation (RAG) app built with:

- **Microsoft Phi-3.5-mini-instruct** (runs on your GPU)
- **LangChain**
- **Chroma** vector store
- **Gradio** UI

Upload your own PDFs, ask questions, and get answers grounded in your documents — with retrieved source chunks shown under every reply. No API keys. Everything runs offline after setup.

## Features

- PDF upload from the UI
- Automatic chunking + embedding
- Persistent Chroma vector database
- Source citations (filename, page, preview)
- Streaming answers
- Reset database button
- Fully local (CUDA GPU)

## Requirements

- NVIDIA GPU with CUDA
- Python 3.10+
- ~8 GB+ VRAM recommended

## Installation

```bash
# CUDA-enabled PyTorch (adjust CUDA version if needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

pip install langchain langchain-community langchain-huggingface chromadb pypdf sentence-transformers gradio transformers
```


# Usage
- Put the project files in a folder and open a terminal there
- Run:
  ```bash 
  python app.py
  ```
- Open the local URL (usually http://127.0.0.1:7860)
- Upload one or more PDFs and add them to the knowledge base.
- Ask your questions in the chatbox
  
# Reset
- Use Reset Vector Database to clear the index. Upload new PDFs again to rebuild the database.

# Project Structure
local-rag-chatbot/
├── app.py           # Main application
├── docs/            # PDF documents
├── chroma_db/       # Vector store (created at runtime)
└── README.md

# How it works
- PDFs are split into ~800-character chunks
- Chunks are embedded with sentence-transformers/all-MiniLM-L6-v2
- Embeddings are stored in Chroma
- On each question, the top matching chunks are retrieved
- Phi-3.5 answers using only that context
- Sources are displayed under the answer

# Notes
- Uses attn_implementation="eager" and use_cache=False for Phi-3.5 compatibility
- First model download can take a few minutes
- Generation speed depends on GPU and prompt length

# Tech Stack
- Phi-3.5-mini-instruct
- LangChain
- Chroma
- Hugging Face Transformers + Sentence Transformers
- Gradio
- PyTorch (CUDA)
