from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
import os
from pathlib import Path

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

def get_vector_db():
    embeddings = OpenAIEmbeddings()
    vector_db = Chroma(persist_directory=get_dbpath(), embedding_function=embeddings)
    return vector_db

def printout_results(answer):
    for doc, score in answer:
        print(f"Score: {score},Content: doc.page_content, Metadata: {doc.metadata}")

def printout_retrieved_docs(docs):
    for doc in docs:
        print(f"Retrieved Document: {doc.page_content}")

def queryCompile(vector_db, query):
    answer = vector_db.similarity_search_with_score(query, k=2)
    retriever = vector_db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "lambda_mult": 0.5}
    )


    # Use the retriever
    docs = retriever.invoke(query)

    return answer, docs

if __name__ == '__main__':
    set_api_env_and_keys()
    vector_db = get_vector_db()
    query = input("Prompt:")
    while  query != 'exit':
        answer, docs = queryCompile(vector_db, query)
        printout_results(answer)
       # printout_retrieved_docs(docs)
        query = input("Next Prompt:")
