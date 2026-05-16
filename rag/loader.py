import os
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader, TextLoader, UnstructuredMarkdownLoader, UnstructuredHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

folder_path = "./Documents"

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
embedding = OpenAIEmbeddings(model="text-embedding-3-small")


def load_folder_pdf(folder_path: str):
    all_doc = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        print(f"Loading: {filename}")
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif filename.endswith(".txt"):
            loader = TextLoader(file_path)
        elif filename.endswith(".md"):
            loader = UnstructuredMarkdownLoader(file_path)
        elif filename.endswith(".html"):
            loader = UnstructuredHTMLLoader(file_path)
        else:
            continue
        docs = loader.load()
        all_doc.extend(docs)
    return all_doc


documents = load_folder_pdf(folder_path)
print(f"Loaded {len(documents)} documents")

chunks = text_splitter.split_documents(documents)

# FAISS.from_documents crashes if chunks is empty so we handle that here
if chunks:
    vector_store = FAISS.from_documents(chunks, embedding)
else:
    print("No documents found in the folder. vector store is empty, add docs via /ingest")
    vector_store = FAISS.from_texts(["placeholder"], embedding)

retriver = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})


def add_documents_from_file(file_path: str):
    filename = os.path.basename(file_path)
    print(f"Ingesting file: {filename}")
    if filename.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif filename.endswith(".txt"):
        loader = TextLoader(file_path)
    elif filename.endswith(".md"):
        loader = UnstructuredMarkdownLoader(file_path)
    elif filename.endswith(".html"):
        loader = UnstructuredHTMLLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {filename}")
    new_docs = loader.load()
    new_chunks = text_splitter.split_documents(new_docs)
    vector_store.add_documents(new_chunks)
    print(f"Added {len(new_chunks)} chunks from file")


def add_documents_from_url(url: str):
    # load the page from the url, chunk it and add to the existing vector store
    print(f"Ingesting url: {url}")
    loader = WebBaseLoader(url)
    new_docs = loader.load()
    new_chunks = text_splitter.split_documents(new_docs)
    vector_store.add_documents(new_chunks)
    print(f"Added {len(new_chunks)} chunks from url")
