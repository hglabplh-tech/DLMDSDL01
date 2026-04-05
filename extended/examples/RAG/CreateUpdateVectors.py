# %% [markdown]
# 
# %%
import kagglehub
import os
import uuid
from pathlib import Path

from lxml.etree import DocumentInvalid
from numpy.f2py.auxfuncs import throw_error
from oauthlib.oauth2.rfc6749.endpoints import metadata
from pypdf import PdfReader
from langchain_chroma import Chroma
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.embeddings.fake import DeterministicFakeEmbedding
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings

from ChatClientOllama import actual_time
from extended.examples.RAG.ChatClientOllama import get_dbpath

CSIZE_CONST = 1800


def CHUNK_SIZE():
    chunk_size = CSIZE_CONST
    chunk_overlap = ((CSIZE_CONST / 100) * 15)
    return chunk_size, chunk_overlap


def get_embedding(key: str):
    if key == 'openai':
        set_api_env_and_keys()
        return OpenAIEmbeddings()
    if key == 'ollama':
        return OllamaEmbeddings(model="llama3")
    elif key == 'huggingface':
        return HuggingFaceEmbeddings()
    else:
        return DeterministicFakeEmbedding(size=4096)


dataset_of_pdf_files_path = kagglehub.dataset_download('manisha717/dataset-of-pdf-files')
print(f"The dataset is loaded to the path: {dataset_of_pdf_files_path}")


# %% [markdown]
# 
# %%

# %%


# %% [markdown]
# 
# %% [markdown]
# 
# %%
def get_app_key():
    fname = 'app_keyid.sec'
    with open(fname) as f:
        app_key = f.read()
        f.close()
        return app_key


def write_index_row(id, path):
    with open(get_dbpath() + "/id_index", 'w') as csv:
        row = id + ',' + path
        csv.writelines([row])


def set_api_env_and_keys():
    app_key = get_app_key()
    os.environ['LANGCHAIN_TRACING_V2'] = 'true'
    os.environ['LANGCHAIN_ENDPOINT'] = 'https://withpersona.com/verify?inquiry-id=inq_NMrSJeR6Aiv2XciLNSjc5qcsv2vn'
    os.environ['LANGCHAIN_API_KEY'] = app_key
    os.environ['OPENAI_API_KEY'] = app_key
    return


def get_dbpath():
    home = Path.home()
    db_path = os.path.join(home, 'chroma_vectordb')
    if not os.path.exists(db_path):
        os.makedirs(db_path)
    return db_path


# %% [markdown]
# 
# %%
def build_vectors(complete_content):
    # 2. Embed and Store in Vector DB (Chroma)
    chunk_size, chunk_overlap = CHUNK_SIZE()
    splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True
    )
    embeddings = get_embedding('fake')
    chunks = splitter.create_documents(complete_content)
    vector_db = SKLearnVectorStore.from_documents(chunks, embedding=embeddings,
                                                  persist_path=get_dbpath(),
                                                  serializer="parquet")
    vector_db.persist()


    return vector_db


# %% [markdown]

# %%
def extract_text_from_pdf(file_path):
    # creating a pdf reader object
    reader = PdfReader(file_path)
    content = ''
    # printing number of pages in pdf file
    page_count = len(reader.pages)
    print(f'Number of pages: {page_count}')
    # getting a specific page from the pdf file
    for index in range(page_count):
        page = reader.pages[index]
        text = page.extract_text()
        content += text
    return content


# %% [markdown]
# 
# %%
def read_all_docs(data_paths):
    # absolute_path = data_path + '/Pdf'
    content_array = []
    for data_path in data_paths:
        filenames = os.listdir(data_path)
        for filename in filenames:
            content_path = os.path.join(data_path, filename)
            print(f"Collecting: {content_path}.... ")
            text = extract_text_from_pdf(content_path)
            content_array.append(text)
    return 0, content_array


def add_documents(complete_content):
    embeddings = get_embedding('fake')
    vector_db = SKLearnVectorStore(embedding=embeddings,
                                   persist_path=get_dbpath(),
                                   serializer="parquet")
    splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True
    )
    documents = []
    ids = []
    for content in complete_content:
        chunks = splitter.split_text(content)
        print(f"Content is added {chunks}")
        doc = Document(page_content=content, metadata={})
        documents.append(doc)
        uu_id = uuid.uuid4()
        print(f"The document ID: {uu_id}")
        ids.append(f"{uu_id}")

    vector_db.add_documents(documents=documents, ids=ids, embedding=embeddings)
    vector_db.persist()
    return vector_db


def update_vectors(complete_content):
    splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True
    )
    embeddings = get_embedding('openai')
    chunks = splitter.split_text(complete_content)
    vector_db = Chroma(persist_directory=get_dbpath(), embedding_function=embeddings)
    Chroma.from_documents(chunks, embedding=embeddings, persist_directory=get_dbpath())
    return vector_db


# %% [markdown]
# 
# %%
# docs_array = read_all_docs(dataset_of_pdf_files_path)
if __name__ == '__main__':
    set_api_env_and_keys()
    absolute_path = dataset_of_pdf_files_path + '/Pdf'
    mode = input("Select mode create / add / update(**later**): ")
    print(f"start collecting pdf datas at {actual_time()}....")
    if mode == 'create':
        ret_code, complete_content = read_all_docs(
            ['/Users/hglabplhak/pdfdb', '/Users/hglabplhak/pdfprivdb', '/Users/hglabplhak/persons'])

        print("build vector")
        vector_db = build_vectors(complete_content)
    if mode == 'add':
        # have to rewrite this
        ret_code2, complete_content2 = read_all_docs(['/Users/hglabplhak/pdfadds'])
        print(complete_content2[0])
        print(f"Count of docs: {len(complete_content2)}")
        chunk_size, chunk_overlap = CHUNK_SIZE()
        print(f"build vector with chunk-size: {chunk_size} and chunk-overlap: {chunk_overlap}")
        vector_db = add_documents(complete_content2)
    print(f"ready at {actual_time()}")
# %% [markdown]
# 
# %%

# %% [markdown]
#
