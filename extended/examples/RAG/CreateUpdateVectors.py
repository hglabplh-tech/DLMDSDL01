# %% [markdown]
# 
# %%
import kagglehub
from pathlib import Path
from pypdf import PdfReader
from langchain_chroma import Chroma
from langchain_text_splitters import CharacterTextSplitter
import os
from langchain_core.embeddings import DeterministicFakeEmbedding

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
    splitter = CharacterTextSplitter(
                chunk_size = 2000,
                chunk_overlap = 180,
                )

    chunks = splitter.create_documents(complete_content)

    # 2. Embed and Store in Vector DB (Chroma)
    #embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0")
    #embeddings = OllamaEmbeddings(model="llama3")
    embeddings = DeterministicFakeEmbedding(size=8192)
    vector_db = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=get_dbpath())
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
    #absolute_path = data_path + '/Pdf'
    docs_array = []
    for data_path in data_paths:
        filenames = os.listdir(data_path)
        for filename in filenames:
            file_path = os.path.join(data_path, filename)
            print(f"Processing {file_path}")
            try:
                text = extract_text_from_pdf(file_path)
                docs_array.append(text)
            except Exception as e:
                print(f"Error on {filename}: {e}")
                return -1, []
    return 0, docs_array

def update_vectors(complete_content):
    splitter = CharacterTextSplitter()
    embeddings = DeterministicFakeEmbedding(size=8192)
    chunks = splitter.create_documents(complete_content)
    vector_db = Chroma(persist_directory=get_dbpath(), embedding_function=embeddings)
    Chroma.from_documents(chunks, embedding=embeddings, persist_directory=get_dbpath())
    return vector_db
# %% [markdown]
# 
# %%
#docs_array = read_all_docs(dataset_of_pdf_files_path)
if __name__ == '__main__':
    set_api_env_and_keys()
    absolute_path = dataset_of_pdf_files_path + '/Pdf'




    print("build vector")
    mode = input("Select mode create / update: ")
    if mode == 'create':
        ret_code, complete_content = read_all_docs(['/Users/hglabplhak/pdfdb']) #, '/Users/hglabplhak/pdfprivdb', '/Users/hglabplhak/persons'])
        print(complete_content[0])
        print(f"Count of docs: {len(complete_content)}")
        print("build vector")
        vector_db = build_vectors(complete_content)
    if mode == 'update':
        #have to rewrite this
        ret_code2, complete_content2 = read_all_docs(['/Users/hglabplhak/persons'])
        print(complete_content2[0])
        print(f"Count of docs: {len(complete_content2)}")
        print("build vector")
        vector_db = update_vectors(complete_content2)
    print('ready')
# %% [markdown]
# 
# %%

# %% [markdown]
# 