# from langchain.llms import LlamaCpp

# def load_llm():
#     model_path = r"C:\Users\SSGhadge2\test\chatbot\models\tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"  # Download this first
    
#     return LlamaCpp(
#         model_path=model_path,
#         n_ctx=4096,
#         n_batch=512,
#         temperature=0.7,
#         max_tokens=500,
#         verbose=True
#     )
import openai
from langchain.chat_models import ChatOpenAI
import os

def load_llm():
    openai.api_key = "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"  # Set your Groq key 
    openai.api_base = "https://api.groq.com/openai/v1"
    
    return ChatOpenAI(
        model_name="llama-3.3-70b-versatile",
        temperature=0.3,
        openai_api_base=openai.api_base,
        openai_api_key=openai.api_key
    )
