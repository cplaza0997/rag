from utils.splitters import recursive_splitter
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyMuPDFLoader,
    TextLoader,
    WebBaseLoader,
)


@recursive_splitter
def web_loader(urls):
    loader = WebBaseLoader(urls)
    return loader.load()


@recursive_splitter
def pdf_loader(pdf):
    loader = PyMuPDFLoader(pdf)
    return loader.load()


@recursive_splitter
def txt_loader(txt):
    loader = TextLoader(txt)
    return loader.load()


@recursive_splitter
def docx_loader(docx):
    loader = Docx2txtLoader(docx)
    return loader.load()

def collect(docs):
    collection = web_loader(urls=docs['html'])
    collection = collection if collection else []
    funcs={'docx':docx_loader, 'pdf':pdf_loader, 'txt':txt_loader}
    for typ, func in funcs.items():
        for source in docs[typ]:
            try:
                doc = funcs[typ](source)
                collection+=doc
            except:
                pass
    return collection