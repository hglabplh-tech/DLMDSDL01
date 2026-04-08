import time
import datetime
import json
from langchain_chroma import Chroma
import os
from pathlib import Path
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_openai.embeddings import OpenAIEmbeddings
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
    os.environ['LANGCHAIN_API_KEY'] = app_key
    os.environ['OPENAI_API_KEY'] = app_key
    return

def get_dbpath():
    home = Path.home()
    db_path = os.path.join(home, 'sklearn_vectordb')
    if not os.path.exists(db_path):
        os.makedirs(db_path)

    db_path = os.path.join(db_path, "sklearn_store")
    return db_path

def get_vector_db():
    embeddings = OpenAIEmbeddings()
    #embeddings = DeterministicFakeEmbedding(size=4096)
    #embeddings = DeterministicFakeEmbedding(size=1024)
    #embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = SKLearnVectorStore(embedding=embeddings,
                                   persist_path=get_dbpath(),
                                   serializer="parquet")
    return vector_db

def printout_results(answer, result, relevance, q_result):
    print(f"Relevance result: {relevance}")
    print(f"Query result scored: {answer}")
    printout_retrieved_docs(q_result)
    print(f"Document content: {result[0].page_content}")

def printout_retrieved_docs(q_result):
    print(f"Retrieved Result: {q_result}")

def print_answer(result):
    print(result)
   # parser = StrOutputParser()
   # cooked = parser.invoke(result)
   #print(cooked)

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = vector_db.similarity_search(query, k=3)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

def queryExecuteRemote(vector_db, query):
   # result = vector_db.similarity_search(query, k=3)
    answer = vector_db.similarity_search_with_score(query, k=3)
    relevance = vector_db.similarity_search_with_relevance_scores(query, k=3)
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
    q_result = retriever.invoke(query)

    return answer, result, relevance, q_result

def actual_time():
    return datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')

def queryExecuteLocal(vector_db, query):
    print(f"Start query at: {actual_time()}")
    result = vector_db.similarity_search(query, k=3)
    answer = vector_db.similarity_search_with_score(query, k=3)
    relevance = vector_db.similarity_search_with_relevance_scores(query, k=3)

    ### The following code is up to now too slow I try it my way
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    # 2. Connect to local LLM (Ollama)
    llm = ChatOllama(model="llama3.1")

    q_result = retriever.invoke(query)
    # 3. Create the chat/query chain
    qaChain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    chain = qaChain | StrOutputParser() # see below invoke call
    # 4. Query your data


    #q_result = []
    #q_result = chain.invoke(query)
    #### end too slow code
    print(f"Finish query at: {actual_time()}")

    return answer, result, relevance, q_result


def inputPrompt(prompt, index):
    query = input(f"{prompt}({index}) : ")
    return  query

if __name__ == '__main__':
    index = 0
    mode = input("Query Mode (openai/local) : ")
    if mode == "openai":
        set_api_env_and_keys(mode)
    vector_db = get_vector_db()

    query = inputPrompt('Prompt', index)
    while  query != 'exit':
        if mode != 'openai' and mode != 'local':
            print(f'Invalid mode (modes : local / openai):{mode} ')
        if mode == "openai":
            answer, result, relevance, q_result = queryExecuteRemote(vector_db, query)
        if mode == "local":
            answer, result, relevance,q_result = queryExecuteLocal(vector_db, query)
        else:
            print('Wrong choice')
            query = 'exit'

        if query != 'exit':
            printout_results(answer, result, relevance, q_result)
            index = index + 1
            query = inputPrompt('Next Prompt', index)