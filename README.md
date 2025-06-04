# Enterprise Document Chatbot

A document-based chatbot application using LangChain, FAISS vector store, and Streamlit.

## Features

- Document processing and vectorization
- Semantic search using FAISS vector store
- Interactive chat interface with Streamlit
- Source document tracking
- Chat history export

## Project Structure

- `app.py` - Main Streamlit application with chat interface
- `document_processor.py` - Document loading and processing utilities
- `vector_store.py` - Vector store management
- `requirements.txt` - Project dependencies

## Setup and Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Place your documents in the `data` directory

3. Process documents: This will take time
```bash
python document_processor.py
```

4. Run the application:
```bash
streamlit run app.py
```

## Components

### Document Processing
- Loads documents from specified directory
- Splits documents into manageable chunks
- Creates embeddings using HuggingFace models
- Stores vectors in FAISS database

### Vector Store
- Uses FAISS for efficient similarity search
- HuggingFace embeddings (e5-base-v2 model)
- Local storage and retrieval

### Chat Interface
- Streamlit-based web interface
- Chat history management
- Source document tracking
- Export chat functionality

## Dependencies

Key dependencies include:
- langchain
- faiss-cpu
- streamlit
- huggingface-hub
- transformers

See `requirements.txt` for complete list of dependencies.
