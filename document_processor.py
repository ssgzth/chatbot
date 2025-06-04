import os
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def process_documents(data_dir="data"):
    # Load documents
    loader = DirectoryLoader(
        data_dir,
        glob="**/*.*",
        use_multithreading=True,
        show_progress=True
    )
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} documents")

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=64,
        length_function=len
    )
    texts = text_splitter.split_documents(documents)
    print(f"📚 Split into {len(texts)} chunks")

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/e5-base-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    print("✅ Embeddings initialized")

    # Create vector store and save locally
    vectordb = FAISS.from_documents(texts, embeddings)
    vectordb.save_local("vector_store")
    print("💾 Vector store saved at './vector_store'")

if __name__ == "__main__":
    process_documents()

# import os
# from langchain_community.document_loaders import DirectoryLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain.docstore.document import Document
# from langchain.schema import Document as LangchainDocument
# from typing import List
# import nltk

# # Download NLTK punkt tokenizer for better sentence splitting (optional but recommended)
# nltk.download('punkt')

# def clean_documents(documents: List[LangchainDocument]) -> List[LangchainDocument]:
#     """Clean document text (e.g., removing noise, unnecessary whitespace)."""
#     cleaned = []
#     for doc in documents:
#         text = doc.page_content.strip().replace("\n", " ").replace("\t", " ")
#         metadata = doc.metadata
#         cleaned.append(Document(page_content=text, metadata=metadata))
#     return cleaned

# def process_documents(data_dir="data", persist_dir="vector_store"):
#     # Step 1: Load documents
#     loader = DirectoryLoader(
#         data_dir,
#         glob="**/*.*",
#         use_multithreading=True,
#         show_progress=True
#     )
#     documents = loader.load()
#     print(f"✅ Loaded {len(documents)} documents")

#     # Step 2: Clean documents
#     documents = clean_documents(documents)
#     print(f"🧹 Cleaned {len(documents)} documents")

#     # Step 3: Split documents into chunks
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=512,
#         chunk_overlap=64,
#         length_function=len,
#         separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
#     )
#     texts = text_splitter.split_documents(documents)
#     print(f"📚 Split into {len(texts)} chunks")

#     # Step 4: Create embeddings
#     embeddings = HuggingFaceEmbeddings(
#         model_name="sentence-transformers/all-MiniLM-L6-v2",
#         model_kwargs={'device': 'cpu'},
#         encode_kwargs={'normalize_embeddings': True}
#     )
#     print("✅ Embeddings initialized")

#     # Step 5: Create vector store and save locally
#     vectordb = FAISS.from_documents(texts, embeddings)
#     vectordb.save_local(persist_dir)
#     print(f"💾 Vector store saved at '{persist_dir}'")

# if __name__ == "__main__":
#     process_documents()

