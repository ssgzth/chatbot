from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

def get_vector_store():
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/e5-base-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vectordb = FAISS.load_local("vector_store", embeddings,allow_dangerous_deserialization=True) 
    return vectordb
