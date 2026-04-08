# %% [markdown]
# 
# %%
import kagglehub
import os
import uuid
import bs4

from pathlib import Path

from lxml.etree import DocumentInvalid
from numpy.f2py.auxfuncs import throw_error
from oauthlib.oauth2.rfc6749.endpoints import metadata
from pypdf import PdfReader
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

from ChatClientOllama import actual_time
from extended.examples.RAG.ChatClientOllama import get_dbpath

CSIZE_CONST = 4096


def CHUNK_SIZE():
    chunk_size = CSIZE_CONST
    chunk_overlap = ((CSIZE_CONST / 100) *  5)
    return chunk_size, chunk_overlap


def get_embedding(key: str):
    if key == 'openai':
        set_api_env_and_keys()
        return OpenAIEmbeddings()
    if key == 'ollama':
        return OllamaEmbeddings(model="llama3.1")
    elif key == 'huggingface':
        return HuggingFaceEmbeddings()
    else:
        return DeterministicFakeEmbedding(size=4096)


dataset_of_pdf_files_path = kagglehub.dataset_download('manisha717/dataset-of-pdf-files')
print(f"The dataset is loaded to the path: {dataset_of_pdf_files_path}")


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

def read_lines(file_path):
    with open(file_path, 'r') as infile:
        lines = infile.readlines()
        return lines


def set_api_env_and_keys():
    app_key = get_app_key()
    os.environ['LANGCHAIN_TRACING_V2'] = 'true'
    os.environ['LANGCHAIN_ENDPOINT'] = 'https://withpersona.com/verify?inquiry-id=inq_NMrSJeR6Aiv2XciLNSjc5qcsv2vn'
    os.environ['LANGCHAIN_API_KEY'] = app_key
    os.environ['OPENAI_API_KEY'] = app_key
    return



# %% [markdown]
# 
# %%
def build_vectors(complete_content):
    # 2. Embed and Store in Vector DB (Chroma)
    chunk_size, chunk_overlap = CHUNK_SIZE()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    embeddings = get_embedding('openai')
    print(f"The complete count of documents is: {len(complete_content)}")
    print(f"The first element is : {complete_content[0]}")
    chunks = splitter.split_documents(complete_content)
    print(f"Split pages post into {len(chunks)} sub-documents.")
    vector_db = SKLearnVectorStore.from_documents(chunks, embedding=embeddings,
                                                  persist_path=get_dbpath(),
                                                  serializer="parquet")
    vector_db.persist()


    return vector_db

def extract_doc_from_web_html(url):
    # Only keep post title, headers, and content from the full HTML.
    bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
    loader = WebBaseLoader(
        web_paths=(url),
        bs_kwargs={"parse_only": bs4_strainer},
    )
    docs = loader.load()
    return docs

def extract_doc_from_html(file_path):
    html_loader = UnstructuredHTMLLoader(file_path, mode='page')
    docs = html_loader.load()
    return docs

def extract_doc_from_text(file_path):
    text_loader = TextLoader(file_path, mode='page')
    docs = text_loader.load()
    return docs

def extract_doc_from_markdown(file_path):
    md_loader = UnstructuredMarkdownLoader(file_path, mode='page')
    docs = md_loader.load()
    return docs

def extract_doc_from_word(file_path):
    docx_loader = UnstructuredWordDocumentLoader(file_path, mode='page')
    docs = docx_loader.load()
    return docs


def extract_doc_from_pdf(file_path):
    # creating a pdf reader object


    loader = PyPDFLoader(
        file_path,
        mode="page",
        #images_parser=RapidOCRBlobParser(),
    )
    documents = loader.load()
    # printing number of pages in pdf file
    page_count = len(documents)
    print(f'Number of pages: {page_count}')
    # getting a specific page from the pdf file
    return documents

def get_suffix(f, suffix: str):
    _,ext = os.path.splitext(f)
    return (ext == '.' + suffix)
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
            if get_suffix(content_path, 'pdf'):
                documents = extract_doc_from_pdf(content_path)
                content_array = content_array + documents
            elif get_suffix(content_path, 'txt'):
                documents = extract_doc_from_text(content_path)
                content_array = content_array + documents
            elif get_suffix(content_path, 'md'):
                documents = extract_doc_from_markdown(content_path)
                content_array = content_array + documents
            elif get_suffix(content_path, 'docx'):
                documents = extract_doc_from_word(content_path)
                content_array = content_array + documents
            elif get_suffix(content_path, 'html') or get_suffix(content_path, 'htm'):
                documents = extract_doc_from_html(content_path)
                content_array = content_array + documents
            elif get_suffix(content_path, 'wbx'):
                lines = read_lines(content_path)
                for url in lines:
                    documents = extract_doc_from_web_html(url)
                    content_array = content_array + documents
            else:
                print(f"The {content_path} cannot pe processed.... go on with next entry")
    return 0, content_array


def add_documents(complete_content):
    embeddings = get_embedding('openai')
    vector_db = SKLearnVectorStore(embedding=embeddings,
                                   persist_path=get_dbpath(),
                                   serializer="parquet")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    print(f"The complete count of documents is: {len(complete_content)}")
    print(f"The first element is : {complete_content[0]}")
    chunks = splitter.split_documents(complete_content)
    print(f"Split pages post into {len(chunks)} sub-documents.")

    vector_db.add_documents(documents=chunks, embedding=embeddings)
    vector_db.persist()
    return vector_db


def update_vectors(complete_content):
    splitter =  RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True
    )
    embeddings = get_embedding('openai')
    chunks = splitter.split_text(complete_content)

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
