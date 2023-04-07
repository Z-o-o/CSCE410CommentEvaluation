import pandas as pd


class Question:
    def __init__(self, question_id, title, answers, tags, body, accepted_answer):
        self.question_id = question_id
        self.title = title
        self.answers = answers
        self.tags = tags
        self.body = body
        self.accepted_answer = accepted_answer


class Answer:
    def __init__(self, answer_id, parent_question_id, body):
        self.answer_id = answer_id
        self.parent_question_id = parent_question_id
        self.body = body


all_python_questions = []
python_db = pd.read_pickle('data/python_database.pkl')
# Remove all questions that have 0 answers
python_db = python_db[python_db.question_answer_count != 0]
current_qa_pointer = python_db.iloc[0]
question = Question(question_id=current_qa_pointer['question_id'], title=current_qa_pointer['question_title'],
                    answers=[], tags=current_qa_pointer['question_tags'], body=current_qa_pointer['question_body'],
                    accepted_answer=None)
for index, row in python_db.iterrows():
    if current_qa_pointer['question_id'] != row['question_id']:
        all_python_questions.append(question)
        question = Question(question_id=row['question_id'],
                            title=row['question_title'],
                            answers=[], tags=row['question_tags'],
                            body=row['question_body'],
                            accepted_answer=None)
    answer = Answer(answer_id=row['answer_id'], parent_question_id=row['answer_parent_id'], body=row['answer_body'])
    try:
        if answer.answer_id == row['question_accepted_answer_id']:
            question.accepted_answer = answer
    except:
        print(f"unable to compare {answer.answer_id} and {row['question_accepted_answer_id']}")
    question.answers.append(answer)
    current_qa_pointer = row
all_python_questions.append(question)

print(all_python_questions[len(all_python_questions)-1].accepted_answer.answer_id)