from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama


def get_embeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
    """
    This function that returns an embedding. An embedding is a vectorial representation of 
    a token.

    Args:
        model_name (str): Name of embedding that comes from Hugging Face.

    Return
        embedding: Embedding.
    """
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings
    

def get_llm():
    """
    Returns Llama3.2 large language model.

    """
    llm = Ollama(model="llama3.2:3b")
    return llm