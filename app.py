from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import gradio as gr
import os

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
    max_new_tokens=300,
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

# Load existing Chroma DB if it exists, otherwise create it
if os.path.exists("chroma_db"):
    print("Loading existing vector store...")
    vectorstore = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )
else:
    print("Creating new vector store from documents...")
    loader = DirectoryLoader(
        "docs",
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} pages")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="chroma_db"
    )

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

template = """Use only the following context to answer the question. 
If you don't know the answer from the context, say you don't know.

Context:
{context}

Question: {question}

Answer:"""

prompt = PromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

def chat(message, history):
    try:
        answer = rag_chain.invoke(message)
        return answer.strip()
    except Exception as e:
        return f"Error: {str(e)}"

demo = gr.ChatInterface(
    fn=chat,
    title="Local RAG Chatbot",
    description="Ask questions about the personality psychology PDFs. Running fully locally with Phi-3.5 + Chroma.",
    examples=[
        "Is personality fixed or can it change?",
        "What did Carl Rogers say about the fully functioning person?",
        "Can repeated behavioral changes alter personality traits?"
    ]
)

print("Launching Gradio interface...")
demo.launch()