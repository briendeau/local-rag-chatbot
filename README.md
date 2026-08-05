# Created By Brian Riendeau Jr. &copy;

# Local RAG Chatbot

A fully local Retrieval-Augmented Generation (RAG) chatbot built with:
- Microsoft Phi-3.5-mini-instruct
- LangChain
- Chroma vector store
- Gradio interface

This project lets you ask questions about your own PDF documents. Everything runs offline on your GPU after the initial setup.

## Features
- Local LLM inference (no API calls)
- PDF document loading and chunking
- Semantic search with Chroma
- Simple Gradio chat interface
- Persistent vector database

## Requirements
- NVIDIA GPU with CUDA
- Python 3.10+
- ~8 GB+ VRAM recommended

## Installation

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install langchain langchain-community langchain-huggingface chromadb pypdf sentence-transformers gradio transformers
```

## Usage
- Put your PDF files in the docs/ folder
- Run the app:
  
```bash
python app.py
```

## Notes
- Uses attn_implementation="eager" for compatibility
- Vector store is saved in the chroma_db/ folder
- This is a portfolio / learning project demonstrating local RAG

## Tech Stack
- Phi-3.5-mini-instruct
- LangChain
- Chroma
- Hugging Face Transformers + Sentence Transformers
- Gradio
- PyTorch (CUDA)

