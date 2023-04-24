import pandas as pd
import json


class Question:
    def __init__(self, question_id, title, answers, tags, body, accepted_answer):
        self.question_id = question_id
        self.title = title
        self.answers = answers
        self.tags = tags
        self.body = body
        self.accepted_answer = accepted_answer


class Answer:
    def __init__(self, answer_id, parent_question_id, body, score):
        self.answer_id = answer_id
        self.parent_question_id = parent_question_id
        self.score = score
        self.label = ""
        self.body = body


def itemize_data(filename):
    all_questions = []
    python_db = pd.read_pickle(filename)
    # Remove all questions that have 0 answers
    python_db = python_db[python_db.question_answer_count > 1]
    current_qa_pointer = python_db.iloc[0]
    question = Question(question_id=current_qa_pointer['question_id'], title=current_qa_pointer['question_title'],
                        answers=[], tags=current_qa_pointer['question_tags'], body=current_qa_pointer['question_body'],
                        accepted_answer=None)
    for index, row in python_db.iterrows():
        if current_qa_pointer['question_id'] != row['question_id']:
            all_questions.append(question)
            question = Question(question_id=row['question_id'],
                                title=row['question_title'],
                                answers=[], tags=row['question_tags'],
                                body=row['question_body'],
                                accepted_answer=None)
        answer = Answer(answer_id=row['answer_id'], parent_question_id=row['answer_parent_id'], score=row['score'], body=row['answer_body'])
        try:
            if answer.answer_id == row['question_accepted_answer_id']:
                question.accepted_answer = answer
        except:
            pass
        question.answers.append(answer)
        current_qa_pointer = row
    all_questions.append(question)
    rank_answers(all_questions)
    get_pos_neg_answers(all_questions)

    return all_questions


def rank_answers(all_questions):
    for question in all_questions:
        question.answers = sorted(question.answers, key=lambda x: x.score, reverse=True)

    return all_questions

def get_pos_neg_answers(all_questions):
    for question in all_questions:
        scores = [x.score for x in question.answers]
        if len(question.answers) > 1:
            upper_percentile = scores[0] * 0.50
            for answer in question.answers:
                if answer.score >= upper_percentile:
                    answer.label = "good"
                else:
                    answer.label = "bad"
                    
def convert_data_to_df(data):
    df_list = []
    for q in data:
        for a in q.answers:
            df_list.append({'text': f"{q.body}-{a.body}", 'label':f"{a.label}"})
        
    df = pd.DataFrame(df_list)
    return df
