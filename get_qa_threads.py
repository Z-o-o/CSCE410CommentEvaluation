import pandas as pd
import json
from torch.utils.data import Dataset
import numpy as np


class Question:
    def __init__(self, question_id, title, answers, tags, body, accepted_answer):
        self.question_id = question_id
        self.title = title
        self.answers = answers
        self.tags = tags
        self.body = body
        self.accepted_answer = accepted_answer
        self.good_answers = []
        self.bad_answers = []


class Answer:
    def __init__(self, answer_id, parent_question_id, body, score):
        self.answer_id = answer_id
        self.parent_question_id = parent_question_id
        self.score = score
        self.body = body


def itemize_data(filename):
    all_questions = []
    python_db = pd.read_pickle(filename)
    # Remove all questions that have 0 answers
    python_db = python_db[python_db.question_answer_count != 0]
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
        answer = Answer(answer_id=row['answer_id'], parent_question_id=row['answer_parent_id'], score=row['score'],
                        body=row['answer_body'])
        try:
            if answer.answer_id == row['question_accepted_answer_id']:
                question.accepted_answer = answer
        except:
            # print(f"unable to compare {answer.answer_id} and {row['question_accepted_answer_id']}")
            continue
        question.answers.append(answer)
        current_qa_pointer = row
    all_questions.append(question)

    return all_questions


def rank_answers(all_questions):
    for question in all_questions:
        question.answers = sorted(question.answers, key=lambda x: x.score, reverse=True)

    return all_questions


def get_pos_neg_answers(all_questions):
    for question in all_questions:
        scores = [x.score for x in question.answers]
        if len(question.answers) > 1:
            upper_percentile = scores[0] * 0.75
            for answer in question.answers:
                if answer.score >= upper_percentile:
                    question.good_answers.append(answer)
                else:
                    question.bad_answers.append(answer)


class QADataset(Dataset):

    def __init__(self, questions):
        self.questions = questions

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, item):
        question = self.questions[item]
        ranked_questions = dict()
        for answer in question.answers:
            ranked_questions[answer.body] = answer.score
        sample = {'question_body': question.body, 'ranked_answers': ranked_questions}
        return sample


python_data = itemize_data('data/python_database.pkl')
ranked_python = rank_answers(python_data)
python_dataset = QADataset(ranked_python)
print(python_dataset.__getitem__(0))
