from langchain.retrievers.multi_query import MultiQueryRetriever


def get_retriever(vdb, llm):
    """
    Returns an retriever object.

    Args:
        vdb (Vector Store): This is a vector database where documents are stored.
        llm (LMM): Large Language model.

    Returns:
        retriever: Retriever.
    """
    retriever = MultiQueryRetriever.from_llm(retriever=vdb.as_retriever(), llm=llm)
    return retriever