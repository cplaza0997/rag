import hashlib
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter
)


def recursive_splitter(func_loader):
    """
    It splits a document into small ones.

    Args:
        func_loader (function): Function that loads a document.

    Returns:
        list: List of chunks of documents.        
    """
    def wrapper(*args, **kwargs):
        doc = func_loader(*args, **kwargs) 
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=50,
            length_function=len,
        )
        chunks = text_splitter.split_documents(doc)
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk"] = i
            chunk.metadata["id"]= hashlib.sha1(chunk.metadata["source"].encode()).hexdigest()+f"_{i}"
        return chunks
    return wrapper