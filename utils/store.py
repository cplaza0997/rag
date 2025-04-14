from langchain.vectorstores import Chroma

def start_db(embeddings):
    vectordb = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    return vectordb

def add_docs(vdb, docs):
    vdb.add_documents(docs, ids=[doc.metadata['id'] for doc in docs])