import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

folder_path = "./Documents"


def load_folder_pdf(folder_path: str):
    all_doc = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            print(f"Loading: {filename}")
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            all_doc.extend(docs)
    return all_doc


documents = load_folder_pdf(folder_path)
print(f"Loaded {len(documents)} documents")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
chunks = text_splitter.split_documents(documents)

embedding = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = FAISS.from_documents(chunks, embedding)

retriver = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
