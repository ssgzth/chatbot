# import streamlit as st
# from vector_store import get_vector_store
# from llm_loader import load_llm
# from langchain.chains import RetrievalQA
# from langchain.prompts import PromptTemplate

# # App title
# st.set_page_config(page_title="Enterprise Document Chatbot", layout="wide")

# @st.cache_resource
# def initialize_components():
#     """Initialize LLM and vector store with caching"""
#     llm = load_llm()
#     vectordb = get_vector_store()
#     return llm, vectordb

# def main():
#     st.title("📁 Document Chatbot Assistant")
    
#     # Load components
#     with st.spinner("Loading AI..."):
#         llm, vectordb = initialize_components()

#     # Custom prompt
#     prompt_template = """You are a document assistant. Use ONLY the context below.

# If unsure, say:
# "I'm sorry, I couldn't find an answer in the documents. Please try asking differently. Thank you!"

# Think step-by-step before answering.

# <context>
# {context}
# </context>

# Question: {question}
# Answer:"""



#     PROMPT = PromptTemplate(
#         template=prompt_template,
#         input_variables=["context", "question"]
#     )

#     qa_chain = RetrievalQA.from_chain_type(
#         llm,
#         chain_type="stuff",
#         retriever=vectordb.as_retriever(search_kwargs={"k": 4}),
#         chain_type_kwargs={"prompt": PROMPT},
#         return_source_documents=True
#     )

#     # Chat history
#     if "messages" not in st.session_state:
#         st.session_state.messages = []

#     # Display previous chat
#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])

#     # New user input
#     if prompt := st.chat_input("Ask a question about your documents..."):
#         st.session_state.messages.append({"role": "user", "content": prompt})
#         with st.chat_message("user"):
#             st.markdown(prompt)

#         # Run QA chain
#         with st.chat_message("assistant"):
#             with st.spinner("Thinking..."):
#                 result = qa_chain({"query": prompt})
#                 response = result["result"].strip()

#                 st.markdown(response)

#                 sources = {doc.metadata.get("source", "Unknown") for doc in result["source_documents"]}
#                 if sources:
#                     st.markdown("---")
#                     st.caption("📄 Sources:")
#                     for source in sources:
#                         st.caption(f"- {source}")

#         st.session_state.messages.append({"role": "assistant", "content": response})

# if __name__ == "__main__":
#     main()


import streamlit as st
from vector_store import get_vector_store
from llm_loader import load_llm
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# App title and layout
st.set_page_config(page_title="Enterprise Document Chatbot", layout="wide")

@st.cache_resource
def initialize_components():
    """Initialize the LLM and vector store with caching"""
    llm = load_llm()
    vectordb = get_vector_store()
    return llm, vectordb

def build_prompt():
    """Returns a custom prompt for the chatbot"""
    template = """
You are a helpful and professional assistant answering based on the provided documents.
ONLY use the context below to answer the question.

If the answer is not in the context, respond with:
"I'm sorry, I couldn't find an answer in the documents. Please try rephrasing your question."

Think step-by-step, use bullet points or structured responses if helpful.

<context>
{context}
</context>

Question: {question}
Answer:"""
    return PromptTemplate(template=template, input_variables=["context", "question"])

def main():
    st.title("📁 Document Chatbot Assistant")

    # Load model and vector database 
    with st.spinner("🔄 Loading AI components..."):
        llm, vectordb = initialize_components()

    # Prompt template
    prompt = build_prompt()

    # RetrievalQA chain setup
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever(search_kwargs={"k": 20}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input from user
    user_query = st.chat_input("Ask a question about your documents...")

    if user_query:
        # Show user's message
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Get LLM response
        with st.chat_message("assistant"):
            with st.spinner("💬 Generating response..."):
                result = qa_chain({"query": user_query})
                response = result["result"].strip()

                st.markdown(response)

                # Source documents
                sources = {doc.metadata.get("source", "Unknown") for doc in result.get("source_documents", [])}
                if sources:
                    st.markdown("---")
                    st.caption("📄 Sources used:")
                    for src in sorted(sources):
                        st.caption(f"- `{src}`")

        # Save assistant response
        st.session_state.messages.append({"role": "assistant", "content": response})

    # Optional: Export chat history
    with st.sidebar:
        st.header("🧾 Chat Options")
        if st.button("🔽 Export Chat"):
            chat_text = "\n\n".join(
                f"{msg['role'].capitalize()}:\n{msg['content']}" for msg in st.session_state.messages
            )
            st.download_button("Download as .txt", chat_text, "chat_history.txt")

if __name__ == "__main__":
    main()
