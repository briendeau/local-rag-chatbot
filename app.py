from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, TextIteratorStreamer
import torch
import gradio as gr
import os
import shutil
import threading
import gc
import time

print("Loading model...")

model_name = "microsoft/Phi-3.5-mini-instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cuda",
    torch_dtype=torch.float16,
    trust_remote_code=True,
    attn_implementation="eager"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=200,
    do_sample=False,
    use_cache=False,
    pad_token_id=tokenizer.eos_token_id,
    return_full_text=False
)

llm = HuggingFacePipeline(pipeline=pipe)

print("Setting up embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ---------- Vector store helpers ----------
def load_or_create_vectorstore():
    if os.path.exists("chroma_db"):
        print("Loading existing vector store...")
        return Chroma(
            persist_directory="chroma_db",
            embedding_function=embeddings
        )
    else:
        print("No existing DB found.")
        if os.path.exists("docs") and any(f.lower().endswith(".pdf") for f in os.listdir("docs")):
            return rebuild_from_docs()
        return None

def rebuild_from_docs():
    if not os.path.exists("docs"):
        os.makedirs("docs")

    pdf_files = [f for f in os.listdir("docs") if f.lower().endswith(".pdf")]
    if not pdf_files:
        return None

    loader = DirectoryLoader(
        "docs",
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    if not documents:
        return None

    print(f"Loaded {len(documents)} pages")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")

    vs = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="chroma_db"
    )
    return vs

def add_pdfs_to_store(pdf_paths):
    """Load only the new PDFs and add them to the existing vector store."""
    global vectorstore

    docs = []
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        docs.extend(loader.load())

    if not docs:
        return 0

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    chunks = text_splitter.split_documents(docs)
    print(f"Adding {len(chunks)} new chunks from {len(pdf_paths)} file(s)")

    if vectorstore is None:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory="chroma_db"
        )
    else:
        vectorstore.add_documents(chunks)

    return len(chunks)

vectorstore = load_or_create_vectorstore()

template = """Use only the following context to answer the question.
If you don't know the answer from the context, say you don't know.

Context:
{context}

Question: {question}

Answer:"""

prompt = PromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def format_sources(docs):
    if not docs:
        return "_No sources retrieved._"
    sources = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "?")
        filename = os.path.basename(source)
        preview = doc.page_content[:160].replace("\n", " ").strip() + "..."
        sources.append(f"**Source {i}:** `{filename}` (page {page})\n> {preview}")
    return "\n\n".join(sources)

# ---------- Core actions ----------
def process_uploads(files):
    global vectorstore
    if not files:
        return "No files uploaded."

    if not os.path.exists("docs"):
        os.makedirs("docs")

    saved_paths = []
    try:
        for f in files:
            src = f if isinstance(f, str) else getattr(f, "name", str(f))
            filename = os.path.basename(src)
            dest = os.path.join("docs", filename)
            shutil.copy(src, dest)
            saved_paths.append(dest)
            print(f"Saved: {dest}")

        n_chunks = add_pdfs_to_store(saved_paths)
        names = [os.path.basename(p) for p in saved_paths]
        return f"Added and indexed: {', '.join(names)} ({n_chunks} chunks)"

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error while indexing: {type(e).__name__}: {e}"

def reset_database():
    global vectorstore

    # Release file handles held by Chroma
    vectorstore = None
    gc.collect()
    time.sleep(0.5)

    if os.path.exists("chroma_db"):
        for attempt in range(3):
            try:
                shutil.rmtree("chroma_db")
                break
            except PermissionError:
                gc.collect()
                time.sleep(1)
        else:
            return "Could not delete chroma_db (file in use). Close the app and delete the folder manually."

    return "Vector database reset. Upload PDFs again to rebuild."

def chat_stream(message, history):
    global vectorstore

    if vectorstore is None:
        yield "No documents loaded. Upload PDFs first, then ask a question."
        return

    if not message or not message.strip():
        yield "Please enter a question."
        return

    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(message)
        context = format_docs(docs)
        sources_text = format_sources(docs)

        chain_input = {"context": context, "question": message}
        prompt_text = prompt.invoke(chain_input).to_string()

        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )

        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=200,
            do_sample=False,
            use_cache=False,
            pad_token_id=tokenizer.eos_token_id
        )

        thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        partial = ""
        for token in streamer:
            partial += token
            yield partial + "\n\n---\n**Retrieved Sources:**\n_Loading..._"

        thread.join()
        yield partial.strip() + "\n\n---\n**Retrieved Sources:**\n" + sources_text

    except Exception as e:
        yield f"Error: {str(e)}"

# ---------- Gradio UI ----------
with gr.Blocks(title="Local RAG Chatbot") as demo:
    gr.Markdown(
        """
        # Local RAG Chatbot
        Fully local Retrieval-Augmented Generation using **Phi-3.5**, **Chroma**, and **LangChain**.

        - Upload your own PDFs
        - Ask questions grounded in your documents
        - See retrieved sources under every answer
        - Everything runs on your GPU (no API keys)
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            file_upload = gr.File(
                label="Upload PDF(s)",
                file_count="multiple",
                file_types=[".pdf"]
            )
            upload_btn = gr.Button("Add to Knowledge Base", variant="primary")
            upload_status = gr.Textbox(label="Upload status", interactive=False)

            reset_btn = gr.Button("Reset Vector Database", variant="stop")
            reset_status = gr.Textbox(label="Reset status", interactive=False)

            gr.Markdown(
                """
                ### Tips
                1. Upload one or more PDFs
                2. Click **Add to Knowledge Base**
                3. Ask questions in the chat
                4. Use **Reset** to clear the index
                """
            )

        with gr.Column(scale=2):
            chatbot = gr.ChatInterface(
                fn=chat_stream,
                examples=[
                    "What is the main idea of this document?",
                    "Summarize the key points.",
                    "What does the author say about change?"
                ]
            )

    upload_btn.click(
        fn=process_uploads,
        inputs=[file_upload],
        outputs=[upload_status]
    )
    reset_btn.click(
        fn=reset_database,
        inputs=[],
        outputs=[reset_status]
    )

print("Launching Gradio interface...")
demo.launch()