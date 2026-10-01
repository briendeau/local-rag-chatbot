# Local RAG from scratch

A small retrieval-augmented generation pipeline that runs **entirely on this machine** via [Ollama](https://ollama.com). No cloud API key required.

This repo is a learning build. The lecture scripts under `aLectures/` are the parts I can walk through line by line: HTTP chat, embeddings, cosine search, grounded generation, and file chunking.

Older Gradio / LangChain / Phi-3 prototypes in the tree were UI experiments. They are **not** the source of truth for how this pipeline works.

## What it does

```
notes/*.txt
    → chunk (sliding window)
    → embed with nomic-embed-text
    → cosine-rank against the question
    → stuff top chunks into a prompt
    → llama3.2 answers only from that context
```

If the notes do not contain the answer, the model is instructed to reply `DON'T KNOW`.

## Stack

| Piece | Choice |
|---|---|
| Runtime | Python 3 + venv |
| Local server | Ollama at `http://localhost:11434/v1` |
| Chat model | `llama3.2` |
| Embedding model | `nomic-embed-text` |
| Client | `openai` Python SDK pointed at localhost |
| Vector store (lectures) | in-memory list + cosine (no Chroma yet) |

## Layout

```
aLectures/
  a1_chat.py          # multi-turn chat against llama3.2
  a2_search.py        # embed notes, rank by cosine
  a3_rag.py           # search + grounded generate (hardcoded notes)
  a4_betterrag.py     # same as A3, notes loaded from files and chunked
  notes/
    json.txt
    ollama.txt
    rag.txt
```

## Setup

```bash
# 1. Ollama app running (or: ollama serve)
ollama pull llama3.2
ollama pull nomic-embed-text

# 2. Python env
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
pip install openai

# 3. Run from aLectures so the notes/ folder resolves
cd aLectures
python a1_chat.py
python a2_search.py
python a4_betterrag.py
```

The OpenAI client uses a dummy key and a local base URL:

```python
OpenAI(api_key="ollama", base_url="http://localhost:11434/v1")
```

## Demo checks (A4)

| Question | Expected |
|---|---|
| `what is json?` | Answer drawn from `notes/json.txt` |
| `when was 9/11?` / Super Bowl | `DON'T KNOW` |

## Design notes

- Chat and embeddings are different Ollama endpoints (`/chat/completions` vs `/embeddings`).
- JSON on the wire is text; parsed values still have types (`int`, `bool`, `null`, objects, arrays).
- Live Python objects are not JSON. Persist a `dict` (`to_dict` / `from_dict`), not the object.
- Chunking uses a sliding window (`size=300`, `overlap=50`) so search returns a passage, not a whole file.


## License

Personal learning project.
