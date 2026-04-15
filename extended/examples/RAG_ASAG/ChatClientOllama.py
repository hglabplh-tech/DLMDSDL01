import os
from langchain.tools import tool
from langchain.chat_models import init_chat_model

from langchain_classic.chains import LLMChain, SimpleSequentialChain
from langchain_classic.chains import RetrievalQA
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.chains import create_retrieval_chain

from utilities.RAGUtils import get_app_key, get_embedding, get_db_temp_path, get_db_history_path, get_dbpath, query_execute, get_vector_db, printout_results

def set_api_env_and_keys(mode):
    app_key = get_app_key()
    os.environ['LANGCHAIN_TRACING_V2'] = 'true'
    os.environ['LANGCHAIN_API_KEY'] = app_key
    os.environ['OPENAI_API_KEY'] = app_key
    return





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

def inputPrompt(prompt, index):
    query = input(f"{prompt}({index}) : ")
    return  query


def load_vector_db():
    global vector_db
    db_to_use = input("Query DB(main/hist/temp) : ")
    db_path = ''
    if db_to_use == 'main':
        db_path = get_dbpath()
    elif db_to_use == 'hist':
        db_path = get_db_history_path()
    else:
        db_path = get_db_temp_path()
    vector_db = get_vector_db(db_path)


if __name__ == '__main__':
    index = 0
    load_vector_db()

    query = inputPrompt('Prompt', index)
    while  query != 'exit':
        answer, result, relevance,q_result = query_execute(vector_db, query)
        if query != 'exit':
            printout_results(answer, result, relevance, q_result)
            index = index + 1
            query = inputPrompt('Next Prompt', index)