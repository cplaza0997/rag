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
print('llm loaded')
vdb = start_db(embedding)
add_docs(vdb, documents)
print('documents added to vector store')

retriever = get_retriever(vdb, llm, search_kwargs={"k": 5})  # Ajusta k para más docs

prompt_template = """Responde la siguiente pregunta **únicamente** usando el contenido de los documentos a continuación. 
Si no encuentras suficiente información en los documentos, responde literalmente: "No encontré información suficiente." 
No inventes ni completes con conocimientos previos.

Pregunta: {question}

Documentos:
{context}

Respuesta en español:"""

from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template,
)


qa = RetrievalQA.from_chain_type(
    llm=llm, 
    #chain_type="stuff", 
    chain_type="map_rerank",  # Mejor combinación de documentos
    retriever=retriever, 
    return_source_documents=True,  # Para mostrar fuentes
    #chain_type_kwargs={"prompt": prompt}
)
qa.verbose = True  # Habilita el modo verbose para ver detalles de la ejecución
llm.verbose = True
retriever.verbose = True



def query(user_query, chat_history):
    print(user_query)
    result = qa.invoke(user_query)
    answer = result['result']
    sources = result.get('source_documents', [])
    sources_text = "\n\n---\n\n".join([doc.page_content[:500] for doc in sources])
    response = answer + "\n\nFuentes:\n" + (sources_text if sources_text else "No hay fuentes disponibles.")
    
    # Gradio espera una lista de listas o tuplas de largo 2
    chat_history = chat_history or []
    chat_history.append([user_query, response])
    
    return chat_history  # Solo una salida, en formato válido

with gr.Blocks() as rag_application:
    chatbot = gr.Chatbot()
    txt = gr.Textbox(label="Input Query", placeholder="Type your question...", lines=2)
    submit_btn = gr.Button("Consultar")  # Botón que faltaba
    clear = gr.Button("Clear")

    txt.submit(query, inputs=[txt, chatbot], outputs=[chatbot])  # Enter
    submit_btn.click(query, inputs=[txt, chatbot], outputs=[chatbot])  # sBotón clic
    clear.click(lambda: [], None, chatbot)


rag_application.launch(server_name="0.0.0.0", server_port=9090)
