from openai.types.shared_params import metadata
from scipy.spatial import distance
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utilities.RAGUtils import get_embedding, get_db_temp_path, build_vectors


from langchain_core.documents import Document


def get_asag_score(student_answer, reference_answer):
    # Convert text to numerical vectors
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([student_answer, reference_answer])

    # Calculate cosine similarity (0 to 1)
    similarity =  cosine_similarity(tfidf[0:1], tfidf[1:2])
    score = similarity[0][0]
    res_sim = round(score, 2)
    return res_sim


def semantic_asag(student_ans, ref_ans, parent):
    query = student_ans
    # Encode both answers into high-dimensional embeddings
    # load model
    docs = []
    doc = Document(id=12, page_content=ref_ans, metadata={"source": "master_solution"})
    docs.append(doc)
    db_path = get_db_temp_path()
    vector_db =  build_vectors(docs, db_path, parent)

    scored_result = vector_db.similarity_search_with_score(query, k=2)
    relevance_result = vector_db.similarity_search_with_relevance_scores(query, k=2)

   # print(f"Scored result: {scored_result}")
   # print(f"Relevance result: {relevance_result}")
    index = 0
    score = 0
    for item in scored_result[0]:
        if index == 1:
            score = item
        index = index + 1

    index = 0
    relevance  = 0
    for item in relevance_result[0]:
        if index == 1:
            relevance = item
        index = index + 1



    ### The following code is up to now too slow I try it my way
    retriever = vector_db.as_retriever(search_kwargs={"k": 1})
    # 2. Connect to local LLM (Ollama)
    q_result = retriever.invoke(query)

    # Compute cosine similarity between the embeddings
    #vector_db.remove()
    return round(score, 2), round(relevance, 2)

def rule_based_grading(student_ans, keywords):
    student_ans = student_ans.lower()
    score = 0
    for word in keywords:
        if word.lower() in student_ans:
            score += 1
    return score / len(keywords)

def get_jaccard_sim(ans, ref):
    # Step 1: Normalize and convert to sets of unique words
    ans_set = set(ref.lower().split())
    ref_set = set(ans.lower().split())
    # Step 2: make intersection for score
    sym_diff = ans_set.symmetric_difference(ref_set)
    score_intersect = ans_set.intersection(ref_set)
    score_union = ans_set.union(ref_set)


    if score_union:
        score = float(len(score_intersect)) / float(len(score_union))
        distance = float(len(sym_diff)) / float(len(score_union))
    else:
        score = 0.0
        distance = 0.0

    # Step 3: make intersection for relevance
    rel_intersect = ref_set.intersection(ans_set)


    if ref_set:
        relevance = float(len(rel_intersect)) / float(len(ref_set))
    else:
        relevance = 0.0

    return round(score,2), round(relevance, 2), round(distance, 2)

## Examples main
if __name__ == '__main__':
    ref = "The mitochondria is the powerhouse of the cell."
    ans = "Mitochondria are the cells powerhouse."
    print(f"Similarity Score: {get_asag_score(ans, ref)}")

    #ref = "Photosynthesis converts light energy into chemical energy."
   # ans = "Plants use sunlight to make food energy."
    #ans = "Red computers are cool"
    score, relevance = semantic_asag(ans, ref, True)
    print(f"Semantic Score: {score:.2f} - Semantic relevance: {relevance:.2f}")


    key_terms = ["Mitochondria", "Powerhouse", "Cell"]
   # student = "Cells have powerhouse organelles."
    print(f"Keyword Score: {rule_based_grading(ans, key_terms)}")

   # ref = "The mitochondria is the powerhouse of the cell"
   # ans = "Mitochondria is a cell powerhouse"
    #ans = "I saw a pink elefant"
    score, relevance, distance = get_jaccard_sim(ans, ref)
    print(f"Jaccard Similarity Score: {score:.2f} - Relevance: {relevance:.2f}")