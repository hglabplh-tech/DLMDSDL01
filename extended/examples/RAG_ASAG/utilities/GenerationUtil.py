import os
from  pathlib import Path
from openai import  OpenAI
from torch.testing._internal import generated

from RAG_ASAG.utilities.RAGUtils import set_api_env_and_keys_in_parent

response = {}

def generate_text(prompt, model="gpt-4"):
    client = OpenAI()

    #  ResponseTextConfigParam
    response = client.responses.create(model= model,  input= prompt)

    return response

if __name__ == '__main__':
    set_api_env_and_keys_in_parent()
    conf_section = input("Configuration / discipline:")
    conf_section = conf_section.strip()
    work_name = input("Work File Name:")
    exam_work = os.path.join(Path.home(), 'examinations', conf_section)
    exam_work_path = os.path.join(exam_work, work_name + '.exam')
    print(f"File  written will be: {exam_work_path}")
    with open(exam_work_path, 'w') as f:
        question_gen = input("Please enter a text for generate a question or 'exit' to leave: ")
        while question_gen != 'exit':
            if question_gen != 'exit':
                generated_question = generate_text(question_gen)
                f.write(generated_question.output_text + '#' )
                print(generated_question.output_text)
                answer_gen =input("Please enter a text for generate the answer 'exit' to leave: ")
                if answer_gen != 'exit':
                    augmented_answer = ''
                    generated_answer = generate_text(answer_gen)
                    manual_answer  = input("Please enter a text for manual (format ([r - replace, a - append]:: text)answer 'exit' to leave: ")
                    manual_answer = manual_answer.strip().split("::")
                    cmd = manual_answer[0].strip()
                    manual_text = manual_answer[1].strip()
                    print(cmd)
                    print(manual_text)
                    if cmd == 'a':
                        augmented_answer = generated_answer.output_text + ' ' +  manual_text
                    elif cmd == 'r':
                        augmented_answer  =  manual_text

                    f.write(augmented_answer + '#')
                    print(augmented_answer)
                    keywords = input("Please enter keywords for this question:")
                    f.write(keywords +'#')
                    points = input("Please enter points(integer) for this question:")
                    print(keywords)
                    f.write(points)
                    print(points)
                    question_gen = input("Please enter a text for generate a question or 'exit' to leave: ")
                    if question_gen != 'exit':
                        f.write('|')


        f.close()
        set_api_env_and_keys_in_parent()