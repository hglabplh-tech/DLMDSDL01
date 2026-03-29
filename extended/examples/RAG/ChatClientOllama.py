import time

from langchain_chroma import Chroma
import os
from pathlib import Path
from langchain_core.embeddings import DeterministicFakeEmbedding
#from langchain_community.embeddings import HuggingFaceEmbeddings
#from langchain.agents import create_agent
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain_ollama import OllamaLLM
from langchain_classic.chains import RetrievalQA

def get_app_key():
    fname = 'app_keyid.sec'
    with open(fname) as f:
        app_key = f.read()
        f.close()
        return app_key

def set_api_env_and_keys(mode):
    app_key = get_app_key()
    os.environ['LANGCHAIN_TRACING_V2'] = 'true'
    if mode == "openai":
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
    embeddings = DeterministicFakeEmbedding(size=8192)
    #embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory=get_dbpath(), embedding_function=embeddings)
    return vector_db

def printout_results(answer):
    for doc, score in answer:
        print(f"Score: {score},Content: doc.page_content, Metadata: {doc.metadata}")

def printout_retrieved_docs(docs):
    for doc in docs:
        print(f"Retrieved Document: {doc.page_content}")



@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = vector_db.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

def queryCompile(vector_db, query):
    answer = vector_db.similarity_search_with_score(query, k=2)
    retriever = vector_db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "lambda_mult": 0.5}
    )
   # client = vector_db._client
    #embeddings = DeterministicFakeEmbedding(size=8192)
    tools = [retrieve_context]
    # If desired, specify custom instructions
    prompt = (query)
    model = init_chat_model("gpt-5.2")

    #agent = create_agent(model, tools, system_prompt=prompt)

    #for event in agent.stream(
     #       {"messages": [{"role": "user", "content": query}]},
     #       stream_mode="values",
    #):
     #   event["messages"][-1].pretty_print()
    # Use the retriever
    docs = retriever.invoke(query)

    return answer, docs

def queryCompileLocal(vector_db, query):
    answer = vector_db.similarity_search_with_score(query, k=2)
    retriever = vector_db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "lambda_mult": 0.5}
    )
    # 2. Connect to local LLM (Ollama)
    llm = OllamaLLM(model="qwen3")
    # 3. Create the chat/query chain
    qa_chain = RetrievalQA.from_chain(llm=llm, chain_type="map_rerank", verbose=True, retriever=retriever)
    # 4. Query your data
    print(f"Start query at: {time.time()}")
    docs = qa_chain.invoke(query)
    print(f"Finish query at: {time.time()}")
    return answer, docs


if __name__ == '__main__':
    index = 0
    mode = input("Query Mode (openai/local) : ")
    if mode == "openai":
        set_api_env_and_keys(mode)
    vector_db = get_vector_db()
    query = input(f"Prompt({index}) : ")
    while  query != 'exit':
        if mode != 'openai' and mode != 'local':
            print(f'Invalid mode (modes : local / openai):{mode} ')
        if mode == "openai":
            answer, docs = queryCompile(vector_db, query)
        if mode == "local":
            answer, docs = queryCompileLocal(vector_db, query)

        printout_results(answer)
        print(docs)
       # printout_retrieved_docs(docs)
        index = index + 1
        query = input(f"Next Prompt({index}) : ")
