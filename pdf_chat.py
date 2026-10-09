# pip install langchain langchain-openai langchain-chroma langchain-huggingface \
#             pypdf sentence-transformers gradio python-dotenv

import os
from uuid import uuid4
from io import BytesIO

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import gradio as gr
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise RuntimeError("GROQ_API_KEY is not set. Add it to your .env file.")

# model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
model    = ChatOpenAI(
    model="openai/gpt-oss-120b",
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1",
    temperature=0,
)

def build_index(pdf_bytes):
    reader = PdfReader(BytesIO(pdf_bytes))
    pages = [
        Document(page_content=page.extract_text() or "", metadata={"page": page_number})
        for page_number, page in enumerate(reader.pages)
    ]
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks   = [
        chunk
        for chunk in splitter.split_documents(pages)
        if chunk.page_content.strip()
    ]
    if not chunks:
        raise ValueError(
            "No extractable text was found in this PDF. "
            "If it contains scanned pages, run OCR on it and upload it again."
        )
    embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db       = Chroma.from_documents(chunks, embedder)         # ③ in-memory store
    return db



prompt = ChatPromptTemplate.from_template("""
You are a helpful PDF assistant. Answer the question using ONLY the context below.
If the context doesn't contain the answer, say "I couldn't find that in the document."
After your answer, list the page numbers you used as: Sources: page X, page Y.

Context:
{context}

Question: {question}
""")

def ask(db, question):
    chunks  = db.similarity_search(question, k=4)
    context = "\n\n".join(
        f"[page {c.metadata['page'] + 1}] {c.page_content}" for c in chunks)
    chain = prompt | model
    return chain.invoke({"context": context, "question": question}).content


indexes = {}

def upload(pdf_bytes, index_id):
    if not pdf_bytes:
        if index_id:
            indexes.pop(index_id, None)
        return "Please upload a PDF first 📄", None
    if index_id:
        indexes.pop(index_id, None)
    try:
        index = build_index(pdf_bytes)
    except ValueError as error:
        return f"Could not index this PDF: {error}", None
    index_id = str(uuid4())
    indexes[index_id] = index
    return "✅ PDF indexed! Ask me anything about it.", index_id

def chat(message, history, index_id):
    if not index_id or index_id not in indexes:
        return "Please upload a PDF first 📄"
    return ask(indexes[index_id], message)

with gr.Blocks(title="📄 Chat with your PDF") as demo:
    gr.Markdown("## 📄 Chat with your PDF (powered by RAG)")
    pdf    = gr.File(label="Upload a PDF", file_types=[".pdf"], type="binary")
    status = gr.Markdown()
    index_state = gr.State(value=None)
    pdf.upload(upload, inputs=[pdf, index_state], outputs=[status, index_state])
    gr.ChatInterface(fn=chat, additional_inputs=[index_state])

demo.launch(share=True)                   # share=True → public link!