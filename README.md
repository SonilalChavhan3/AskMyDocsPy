# AskMyDocsPy

AskMyDocsPy is a simple PDF Q&A application built with Python, LangChain, and Gradio. It lets you upload a PDF, split it into searchable chunks, embed those chunks using Hugging Face models, and then ask questions about the document in natural language.

## Features

- Upload a PDF file
- Extract text from each page using PyPDF
- Split the document into manageable chunks
- Generate vector embeddings with Hugging Face sentence-transformers
- Store the embeddings in Chroma for similarity search
- Ask questions using a chat model through Groq/OpenAI-compatible API
- Receive answers grounded in the document contents

## Tech Stack

- Python
- LangChain
- PyPDF
- Hugging Face Embeddings
- Chroma
- Gradio
- Groq API

## Project Structure

- `pdf_chat.py` — main application code
- `.env` — stores environment variables such as the API key

## Setup

1. Create a virtual environment if needed.
2. Install dependencies:

```bash
pip install langchain langchain-openai langchain-chroma langchain-huggingface pypdf sentence-transformers gradio python-dotenv
```

3. Create a `.env` file in the project root and add your API key:

```env
GROQ_API_KEY=your_api_key_here
```

## Run the App

```bash
python pdf_chat.py
```

Then open the Gradio link in your browser, upload a PDF, and start asking questions.

## How It Works

The app:

1. Loads the uploaded PDF
2. Splits the document into chunks
3. Embeds each chunk
4. Stores those embeddings in a local vector database
5. Finds the most relevant chunks for a question
6. Sends the retrieved context to the LLM to generate a grounded answer

## Notes

- The app expects a valid `GROQ_API_KEY` to be present in the `.env` file.
- PDFs must contain selectable text. Scanned/image-only PDFs need OCR before upload.
- The project is intended for local experimentation and lightweight document Q&A use cases.
- The Gradio app uses a public share link by default (`share=True`), so be careful when using it in a shared or production environment.
