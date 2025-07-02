from langchain.retrievers.multi_query import MultiQueryRetriever

def get_retriever(vdb, llm, search_kwargs=None):
    """
    Returns a retriever object.

    Args:
        vdb (Vector Store): Vector database.
        llm (LLM): Large Language Model.
        search_kwargs (dict, optional): kwargs para la búsqueda en el vector DB (ej: {"k": 5})

    Returns:
        retriever: Retriever.
    """
    retriever = MultiQueryRetriever.from_llm(
        retriever=vdb.as_retriever(search_kwargs=search_kwargs),
        llm=llm
    )
    return retriever