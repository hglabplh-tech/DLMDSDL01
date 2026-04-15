from configparser import ConfigParser
import os
from pathlib import Path

from utilities.ConfigReader import ConfigReader
from utilities.RAGUtils import get_rag_config_path
from utilities.ASAGUtils import get_asag_score, semantic_asag, get_jaccard_sim, rule_based_grading

config = None
def read_exam_solution(exam_file):
    questions = []
    answers = []
    keywords = []
    points_array = []
    tot_points = 0
    with open(exam_file) as f:
        content = f.read()
        splitted_cont = content.split('|')
        for item in splitted_cont:
            splitted_item = item.split('#')
            question = splitted_item.__getitem__(0)
            answer = splitted_item.__getitem__(1)
            keyword = splitted_item.__getitem__(2)
            points = splitted_item.__getitem__(3)
            questions.append(question)
            answers.append(answer)
            keywords.append(keyword)
            int_points = int(points)
            tot_points = tot_points + int_points
            points_array.append(int_points)
    return questions, answers, keywords, points_array, tot_points

def do_score(stud_answer, ref, keywords):
    config = ConfigReader.myinstance(get_rag_config_path(), conf_section)
    asag_semantic_weight = config.read_val_float('asag_semantic_weight')
    asag_jaccard_weight = config.read_val_float('asag_jaccard_weight')
    rule_grad = rule_based_grading(stud_answer, keywords)
    tdiff_scoring = get_asag_score(stud_answer, ref)
    score, relevance = semantic_asag(stud_answer, ref, False)
    score_jac, relevance_jac, distance_jac, rel_dist_jac = get_jaccard_sim(stud_answer, ref)
    tot_rel = ((relevance * asag_semantic_weight) + (relevance_jac * asag_jaccard_weight)) / (asag_jaccard_weight + asag_semantic_weight)
    tot_score = (((score * asag_semantic_weight) + (score_jac * asag_jaccard_weight))) / (asag_jaccard_weight + asag_semantic_weight)

    return score, relevance, tot_score, tot_rel, score_jac, relevance_jac,distance_jac, rel_dist_jac, rule_grad, tdiff_scoring

def calculate_points_per_item(tot_score, tot_rel, rule_grad, tdiff_scoring, points):
    config = ConfigReader.myinstance(get_rag_config_path(), conf_section)
    asag_base_score_limit = config.read_val_float('asag_base_score_limit')
    asag_keywords_limit = config.read_val_float('asag_keywords_limit')
    if rule_grad >= asag_keywords_limit and tdiff_scoring >= asag_base_score_limit and tot_score <= 0.19:
        calc_points = tot_rel * points
    else:
        calc_points = 0.0
    return round(calc_points, 1)

def score_answers(questions, answers, keyword_array, points_array, tot_points,stud_ans, mode):
    tot_calc_points = 0
    for index in range(len(questions)):
        if mode == 'batch':
            stud_answer = stud_ans[index]
        else:
            stud_answer = input(f"Answer the question: {questions[index]}")
        print(f'stud_answer: {stud_answer}')
        answer = answers[index]
        keywords = keyword_array[index]
        points = points_array[index]
        score, relevance, tot_score, tot_rel, score_jac, relevance_jac,dist_jac, rel_dist_jac, rule_grad, tdiff_scoring = do_score(stud_answer, answer, keywords)
        print(f"rule grad: {rule_grad}")
        print(f"ASAG score: {tdiff_scoring}")
        print(f"GPT score: {score}, relevance: {relevance}")
        print(f"Jaccard score: {score_jac}, relevance: {relevance_jac} Distance: {dist_jac} Relevance Dist {rel_dist_jac}")
        print(f"Total score: {tot_score}, relevance: {tot_rel}")
        calc_points = calculate_points_per_item(tot_score, tot_rel, rule_grad, tdiff_scoring, points)
        print(f"Calculate Points Task {index} = {calc_points}")
        tot_calc_points = tot_calc_points + calc_points
    print(f"Total reached points: {round(tot_calc_points, 1)} of {tot_points}")
    percent = ((tot_calc_points  * 100) / tot_points)
    return round(percent, 0)

def set_config(conf_section):
    ConfigReader.myinstance(get_rag_config_path(), conf_section)
    return config

def read_exam_work(path, fname):
    full_path = os.path.join(path, fname)
    answer_array = []
    with open(full_path) as f:
        lines = f.read()
        answers = lines.split('#')
        for index in range(len(answers)):
            item = answers.__getitem__(index)
            answer_array.append(item)
    return answer_array


if __name__ == '__main__':
    conf_section = input("Configuration / discipline:")
    exam_work_path = f'/Users/hglabplhak/examinations/{conf_section}/works/'
    set_config(conf_section)
    mode = input("Mode interactive/batch:")
    if mode == 'batch':
        fname = input('Class work:')
        answer_array = read_exam_work(exam_work_path, fname)
    else:
        answer_array = []
    questions, answers, keywords, points_array, tot_points = read_exam_solution(
        f'/Users/hglabplhak/examinations/{conf_section}/short_test.exam')
    print(questions)
    print(answers)
    print (keywords)
    print(points_array)
    print(tot_points)
    percent =  score_answers(questions, answers, keywords, points_array, tot_points, answer_array, mode)
    print(f"Quit with {percent} %")