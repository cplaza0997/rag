from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM


def get_embeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
    """
    Returns an embedding model from HuggingFace.
    """
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings


def get_llm():
    """
    Returns Llama3.2 large language model via Ollama.
    """
    # deepseek-r1:1.5b
    # llm = OllamaLLM(model="llama3.2:3b")
    llm = OllamaLLM(model="deepseek-r1:1.5b")
    return llm
