import os
import pickle
import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from transformers import pipeline

_translator = None
_embeddings_model = None

def translate_query(query: str) -> str:
    global _translator
    if _translator is None:
        _translator = pipeline("translation", model="Helsinki-NLP/opus-mt-ROMANCE-en")
    
    print("Traduzindo a pergunta...")
    translated_query = _translator(query, max_length=4000)[0]['translation_text']
    print(f"Pergunta traduzida: {translated_query}")
    return translated_query

def load_documents(file_path: str) -> list[Document]:
    cache_path = f"{file_path}.cache.pkl"
    
    if os.path.exists(cache_path) and os.path.getmtime(cache_path) > os.path.getmtime(file_path):
        print("Carregando do cache...")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    
    print("Carregando documento...")
    doc = pymupdf.open(file_path)
    docs = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()

        if text.strip():
            docs.append(Document(
                page_content = text,
                metaData = {
                    "source": file_path,
                    "page": page_num + 1,
                    "total_pages": len(doc)
                }
            ))
    doc.close()
    print(f"{len(docs)} paginas carregadas.")
    
    with open(cache_path, 'wb') as f:
        pickle.dump(docs, f)

def split_documents_into_chunks(docs: list[Document]) -> list[Document]:
    print("Dividindo documentos em chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=800,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(docs)

def creating_embeddings_model() -> HuggingFaceEmbeddings:
    global _embeddings_model
    if _embeddings_model is None:
        print("Criando modelo de embeddings...")
        _embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return _embeddings_model

def create_or_load_vectordb(chunks: list[Document], embeddings_model: HuggingFaceEmbeddings, db_directory: str) -> Chroma: 
    if os.path.exists(db_directory) and len(os.listdir(db_directory)) > 0:
        print("Banco de dados encontrado, carregando...")
        return Chroma(
            persist_directory=db_directory,
            embedding_function=embeddings_model
        )
    else:
        print("Banco de dados não encontrado, criando e salvando...")
        
        if len(chunks) > 100:
            print(f"Processando {len(chunks)} chunks em batches...")
            db = Chroma.from_documents(
                documents=chunks[:100],
                embedding=embeddings_model,
                persist_directory=db_directory
            )
            
            for i in range(100, len(chunks), 100):
                batch = chunks[i:i+100]
                print(f"Batch {i//100 + 1}")
                db.add_documents(batch)
            return db
        else:
            return Chroma.from_documents(
                documents=chunks,
                embedding=embeddings_model,
                persist_directory=db_directory
            )

if __name__ == "__main__":
    pdf_path = './data/DSM-5-TR.pdf'
    db_directory = './chroma_db'

    docs = load_documents(pdf_path)
    chunks = split_documents_into_chunks(docs)
    embeddings_model = creating_embeddings_model()
    db = create_or_load_vectordb(chunks, embeddings_model, db_directory)

    print("Banco de dados pronto para a busca!")

    retriever = db.as_retriever(search_kwargs={"k": 5})
    pergunta_portugues = ("quais sao os transtornos mentais em criancas?")
    pergunta_traduzida = translate_query(pergunta_portugues)
    docs_relevantes = retriever.invoke(pergunta_traduzida)

    for doc in docs_relevantes:
        print(f"conteudo: {doc.page_content}\n")
        print(f"Fonte: {doc.metadata['source']}\n")
        print("=-"*50)