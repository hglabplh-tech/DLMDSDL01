import os
from pathlib import Path
from langchain_chroma import Chroma
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_community.document_loaders import PyPDFLoader, UnstructuredHTMLLoader, WebBaseLoader, TextLoader, UnstructuredMarkdownLoader, UnstructuredWordDocumentLoader
from langchain_community.document_loaders.parsers import RapidOCRBlobParser
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.embeddings.fake import DeterministicFakeEmbedding
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings

CSIZE_CONST = 1024


def CHUNK_SIZE():
    chunk_size = CSIZE_CONST
    chunk_overlap = ((CSIZE_CONST / 100) *  15)
    return chunk_size, chunk_overlap

def get_app_key():
    fname = 'app_keyid.sec'
    with open(fname) as f:
        app_key = f.read()
        f.close()
        return app_key

def get_app_key_in_parent():
    fname = '../app_keyid.sec'
    with open(fname) as f:
        app_key = f.read()
        f.close()
        return app_key

def build_vectors(complete_content, db_path, parent):
    # 2. Embed and Store in Vector DB (Chroma)
    chunk_size, chunk_overlap = CHUNK_SIZE()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    embeddings = get_embedding('openai', parent)
    print(f"The complete count of documents is: {len(complete_content)}")
    print(f"The first element is : {complete_content[0]}")
    chunks = splitter.split_documents(complete_content)
    print(f"Split pages post into {len(chunks)} sub-documents.")
    vector_db = SKLearnVectorStore.from_documents(chunks, embedding=embeddings,
                                                  persist_path=db_path,
                                                  serializer="parquet")
    vector_db.persist()


    return vector_db

def add_documents(complete_content, db_path, parent):
    embeddings = get_embedding('openai', parent)
    chunk_size, chunk_overlap = CHUNK_SIZE()
    vector_db = SKLearnVectorStore(embedding=embeddings,
                                   persist_path=db_path,
                                   serializer="parquet")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    print(f"The complete count of documents is: {len(complete_content)}")
    print(f"The first element is : {complete_content[0]}")
    chunks = splitter.split_documents(complete_content)
    print(f"Split documents into {len(chunks)} sub-documents.")
    print(f"build vector with chunk-size: {chunk_size} and chunk-overlap: {chunk_overlap}")

    vector_db.add_documents(documents=chunks, embedding=embeddings)
    vector_db.persist()
    return vector_db

def get_db_base_path():
    home = Path.home()
    db_base_path = os.path.join(home, 'sklearn_vectordb')
    if not os.path.exists(db_base_path):
        os.makedirs(db_base_path)
    return db_base_path

def get_dbpath():
    home = Path.home()
    db_path = os.path.join(get_db_base_path(), "sklearn_store")
    return db_path

def get_db_temp_path():
    db_path = os.path.join(get_db_base_path(), "temp_store")
    return db_path

def get_db_history_path():
    db_path = os.path.join(get_db_base_path(), "history_edu_store")
    return db_path


def set_api_env_and_keys():
    app_key = get_app_key()
    os.environ['LANGCHAIN_TRACING_V2'] = 'true'
    os.environ['LANGCHAIN_API_KEY'] = app_key
    os.environ['OPENAI_API_KEY'] = app_key
    return

def set_api_env_and_keys_in_parent():
    app_key = get_app_key_in_parent()
    os.environ['LANGCHAIN_TRACING_V2'] = 'true'
    os.environ['LANGCHAIN_API_KEY'] = app_key
    os.environ['OPENAI_API_KEY'] = app_key
    return



def get_embedding(key: str, parent: bool):
    if key == 'openai':
        if parent:
            set_api_env_and_keys_in_parent()
        else:
            set_api_env_and_keys()
        return OpenAIEmbeddings()
    if key == 'ollama':
        return OllamaEmbeddings(model="llama3.1")
    elif key == 'huggingface':
        return HuggingFaceEmbeddings()
    else:
        return DeterministicFakeEmbedding(size=4096)