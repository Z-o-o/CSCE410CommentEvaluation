import json
<<<<<<< HEAD
from transformers import BertTokenizer, logging
import torch

import custom_dataset
import model
=======

from transformers import BertTokenizer
import torch
import argparse
import custom_dataset
import model
import get_qa_threads
>>>>>>> cb3a873c051f289f88d3e7a8027fe175d41a0176


def print_formatted_ranked_answers(ranked, predictions, question):
    print(f'Ranked answers for given question "{question}": ')
    for answer in ranked:
<<<<<<< HEAD
        print(
            f'{ranked.index(answer) + 1}: {answer} ({"Good Answer" if predictions[answer][0] else "Bad Answer"} with '
            f'{"{:.1f}".format(predictions[answer][1] * 100)}% confidence)'
        )


def get_ranked_answers(custom_model, question, answers):
    good_answers = 0
    predictions = dict()
    for answer in answers:
        label, confidence_level = predict(custom_model, f"{question}:{answer}")
        if label:
            good_answers += 1
        predictions[answer] = (label, confidence_level)
    ranked = sorted(
        predictions,
        key=lambda x: (predictions[x][0], predictions[x][1]),
        reverse=True,
    )
    bad_answers = ranked[good_answers:]
    bad_answers.reverse()
    ranked = ranked[:good_answers] + bad_answers
    return ranked, predictions
=======
        print(f'{ranked.index(answer) + 1}: {answer} ({"Good Answer" if predictions[answer][0] else "Bad Answer"} with '
              f'{"{:.1f}".format(predictions[answer][1] * 100)}% confidence)')


def get_ranked_answers(question, answers, predictions):
    good_answers = 0
    for answer in answers:
        label, confidence_level = predict(custom_model, f'{question}:{answer}')
        if label:
            good_answers += 1
        predictions[answer] = (label, confidence_level)
    ranked = sorted(answer_predictions, key=lambda x: (predictions[x][0], predictions[x][1]),
                    reverse=True)
    bad_answers = ranked[good_answers:]
    bad_answers.reverse()
    ranked = ranked[:good_answers] + bad_answers
    return ranked
>>>>>>> cb3a873c051f289f88d3e7a8027fe175d41a0176


def predict(model, text):
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    text_dict = tokenizer(
        text, padding="max_length", max_length=512, truncation=True, return_tensors="pt"
    )
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        model = model.cuda()
    mask = text_dict["attention_mask"].to(device)
    input_id = text_dict["input_ids"].squeeze(1).to(device)
    with torch.no_grad():
        output = model(input_id, mask)
        probs = torch.nn.functional.softmax(output, dim=1)
        conf, classes = torch.max(probs, 1)
        label_id = output.argmax(dim=1).item()
        for key in custom_dataset.labels.keys():
            if custom_dataset.labels[key] == label_id:
                return label_id, conf.item()


<<<<<<< HEAD
def main(filename):
    logging.set_verbosity_error()
    json_file = open(filename)
    json_data = json.load(json_file)
    custom_model = model.BertClassifier()
    custom_model.load_state_dict(torch.load("test-model.pth"))
    ranked_answers, answer_predictions = get_ranked_answers(
        custom_model, json_data["question"], json_data["answers"]
    )
    print_formatted_ranked_answers(
        ranked_answers, answer_predictions, json_data["question"]
    )


if __name__ == "__main__":
    main("example_input.json")
=======
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", "-t")


if __name__ == "__main__":
    json_file = open('example_input.json')
    json_data = json.load(json_file)
    custom_model = model.BertClassifier()
    custom_model.load_state_dict(torch.load("test-model.pth"))
    answer_predictions = dict()
    ranked_answers = get_ranked_answers(json_data['question'], json_data['answers'], answer_predictions)
    print_formatted_ranked_answers(ranked_answers, answer_predictions, json_data['question'])
>>>>>>> cb3a873c051f289f88d3e7a8027fe175d41a0176
