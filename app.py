import gradio as gr
from langchain.chains import RetrievalQA
from utils.loaders import collect
from utils.utils import read_yml
from utils.llm import get_embeddings, get_llm
from utils.store import start_db, add_docs
from utils.retrievers import get_retriever

docs = read_yml('files.yml')

documents = collect(docs)

embedding = get_embeddings()
llm = get_llm()

vdb=start_db(embedding)
add_docs(vdb, documents)

retriever = get_retriever(vdb, llm)

qa = RetrievalQA.from_chain_type(llm=llm, 
                                chain_type="stuff", #"map_rerank"
                                retriever=retriever, 
                                return_source_documents=False)

def query(query):
    response = qa.invoke(query)
    return response['result']

# Create Gradio interface
rag_application = gr.Interface(
    fn=query,
    allow_flagging="never",
    inputs=[
        #gr.File(label="Upload PDF File", file_count="single", file_types=['.pdf'], type="filepath"),  # Drag and drop file upload
        gr.Textbox(label="Input Query", lines=2, placeholder="Type your question...")
    ],
    outputs=gr.Textbox(label="Output"),
    title="RAG Chatbot",
    description="RAG System"
)

# Launch the app
rag_application.launch(server_name="0.0.0.0", server_port= 9090)