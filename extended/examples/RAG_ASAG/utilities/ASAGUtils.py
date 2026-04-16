import os

from openai.types.shared_params import metadata
from scipy.spatial import distance
from sentence_transformers.sentence_transformer.modules import tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utilities.RAGUtils import get_embedding, get_db_temp_path, build_vectors
from langchain_core.documents import Document
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sentence_transformers import SentenceTransformer, util

def get_BERT_PreReq():
    # 1. Initialize tokenizer and model for regression (outputting a score)
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=1) # 1 for regression
    return tokenizer, model

# 2. Prepare input (combining question, reference, and student answer)
def prepare_input(question, reference, student, tokenizer, model):
    return tokenizer(f"{question} [SEP] {reference}", student,
                     truncation=True, padding='max_length', return_tensors="pt")

def get_sentence_scoring(student_answer, reference_answer):


    # 1. Load a pre-trained SBERT model
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')



    # 2. Compute embeddings
    ref_embedding = model.encode(reference_answer, convert_to_tensor=True)
    stu_embedding = model.encode(student_answer, convert_to_tensor=True)

    # 3. Calculate cosine similarity (the "grade")
    score = util.pytorch_cos_sim(ref_embedding, stu_embedding)
    print(f"Grading Score: {score.item():.4f}")
    return score.item()

def get_bert_model_scoring(question, student_answer, reference_answer):
    vectorizer = TfidfVectorizer()
    # Example usage for inference
    tfidf = vectorizer.fit_transform([question, student_answer, reference_answer])
    tokenizer, model = get_BERT_PreReq()
    inputs = prepare_input(question, reference_answer, student_answer,tokenizer, model)
    outputs = model(**inputs)
    predicted_score = outputs.logits.item()
    return predicted_score


def doErnieAndBERT(answer='NO Answer given'):
    # 1. Load the tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # 2. Tokenize and convert to IDs
    inputs = tokenizer(answer, return_tensors="pt")
    print(inputs['input_ids'])
    return inputs

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
        if index == 0:
            answer = item.page_content
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
    return round(score, 2), round(relevance, 2), answer

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
        score_distance = float(len(sym_diff)) / float(len(score_union))
    else:
        score = 0.0
        score_distance = 0.0

    # Step 3: make intersection for relevance
    rel_intersect = ref_set.intersection(ans_set)


    if ref_set:
        relevance = float(len(rel_intersect)) / float(len(ref_set))
        rel_distance =  float(len(sym_diff)) / float(len(ref_set))
    else:
        relevance = 0.0
        rel_distance  = 0.0

    return round(score,2), round(relevance, 2), round(score_distance, 2), round(rel_distance, 2)

## Examples main
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['HF_TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['HF_TOKEN'] ="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"
    ref = "The mitochondria is the powerhouse of the cell."
    ans = "Mitochondria are the cells powerhouse."
    print(f"Similarity Score: {get_asag_score(ans, ref)}")
    question = "Explain what the  Mitochondria is ?"
    #ref = "Photosynthesis converts light energy into chemical energy."
   # ans = "Plants use sunlight to imake food energy."
    #ans = "Red computers are cool"
    score, relevance, answer = semantic_asag(ans, ref, True)
    print(f"Semantic Score: {score:.2f} - Semantic relevance: {relevance:.2f}")
    doErnieAndBERT(answer)
    sentence_score = get_sentence_scoring(ans, ref)
    print(f"sentence Score: {sentence_score:.2f}")
    bert_model_score = get_bert_model_scoring(question, ans, ref)
    print(f"bert model Score: {bert_model_score:.2f}")

    key_terms = ["Mitochondria", "Powerhouse", "Cell"]
   # student = "Cells have powerhouse organelles."
    print(f"Keyword Score: {rule_based_grading(ans, key_terms)}")

   # ref = "The mitochondria is the powerhouse of the cell"
   # ans = "Mitochondria is a cell powerhouse"
    #ans = "I saw a pink elefant"
    score, relevance, distance, rel_dist = get_jaccard_sim(ans, ref)
    print(f"Jaccard Similarity Score: {score:.2f} - Relevance: {relevance:.2f}")