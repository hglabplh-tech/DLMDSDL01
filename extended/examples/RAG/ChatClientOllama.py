import time
import datetime
import json
from langchain_chroma import Chroma
import os
from pathlib import Path
from langchain_core.embeddings import DeterministicFakeEmbedding
#from langchain_community.embeddings import HuggingFaceEmbeddings
#from langchain.agents import create_agent
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_ollama import OllamaLLM, OllamaEmbeddings, ChatOllama
from langchain_classic.chains import LLMChain, SimpleSequentialChain
from langchain_classic.chains import RetrievalQA
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.chains import create_retrieval_chain

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
    embeddings = DeterministicFakeEmbedding(size=4096)
    #embeddings = DeterministicFakeEmbedding(size=1024)
    #embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = SKLearnVectorStore(embedding=embeddings,
                                   persist_path=get_dbpath(),
                                   serializer="parquet")
    return vector_db

def printout_results(answer):
    for doc, score in answer:
        print(f"Score: {score},Metadata: {doc.metadata}")

def printout_retrieved_docs(docs):
    for doc in docs:
        print(f"Retrieved Document: {doc.page_content}")

def print_answer(result):
    print(result)
    parser = StrOutputParser()
    cooked = parser.invoke(result)
    print(cooked)

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
    #embeddings = DeterministicFakeEmbedding(size=1024)
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

def actual_time():
    return datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')

def queryCompileLocal(vector_db, query):
    answer = vector_db.similarity_search_with_score(query, k=2)
    retriever = vector_db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "lambda_mult": 0.5}
    )
    # 2. Connect to local LLM (Ollama)
    llm = ChatOllama(model="llama3")

    # 3. Create the chat/query chain
    qaChain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    chain = qaChain | StrOutputParser() # see below invoke call
    # 4. Query your data
    print(f"Start query at: {actual_time()}")
    docs = chain.invoke(query) # TODO: the chain thing does not work in cause of the parser have to write a own JSON parsing
    #docs = qaChain.invoke(query) # was invoke before
    print(f"Finish query at: {actual_time()}")
    return answer, docs


def inputPrompt(prompt, index):
    area = input(f"Area({index}) : ")
    query = input(f"{prompt}({index}) : ")
   # prompt  = area + ' ' +  query
    prompt = query # temporarily
    return prompt, query

if __name__ == '__main__':
    index = 0
    mode = input("Query Mode (openai/local) : ")
    if mode == "openai":
        set_api_env_and_keys(mode)
    vector_db = get_vector_db()

    prompt, query = inputPrompt('Prompt', index)
    while  query != 'exit':
        if mode != 'openai' and mode != 'local':
            print(f'Invalid mode (modes : local / openai):{mode} ')
        if mode == "openai":
            answer, docs = queryCompile(vector_db, prompt)
        if mode == "local":
            answer, docs = queryCompileLocal(vector_db, prompt)

        printout_results(answer)
        print_answer(docs)
       # printout_retrieved_docs(docs)
        index = index + 1
        prommpt, query = inputPrompt('Next Prompt', index)
