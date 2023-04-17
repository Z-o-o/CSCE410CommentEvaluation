from transformers import BertPreTrainedModel, BertForSequenceClassification, BertTokenizer, AdamW
import torch
from torch.nn.functional import cross_entropy
from get_qa_threads import itemize_data, rank_answers, QADataset, get_pos_neg_answers
import pickle


class MonoBERT(BertPreTrainedModel):
    def __init__(self, config):
        config.num_labels = 1
        super(MonoBERT, self).__init__(config)
        self.bert = BertForSequenceClassification(config)
        self.init_weights()

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        logits = outputs[0]
        return logits


model = MonoBERT.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

optimizer = AdamW(model.parameters(), lr=1e-5, eps=1e-8)
optimizer.zero_grad()

all_questions = itemize_data('data/python_database.pkl')
rank_answers(all_questions)
get_pos_neg_answers(all_questions)
dataset = QADataset(all_questions)

j = 0
print(len(dataset.questions))
for question in dataset.questions:
    print(j)
    for i in range(min(len(question.good_answers), len(question.bad_answers))):
        pos_text = "{} [SEP] {}".format(question.body, question.good_answers[i].body)
        neg_text = "{} [SEP] {}".format(question.body, question.bad_answers[len(question.bad_answers) - 1 - i].body)

        pos_encoded = tokenizer.encode_plus(pos_text, return_tensors="pt", padding=True, truncation=True,max_length=500, add_special_tokens = True)
        neg_encoded = tokenizer.encode_plus(neg_text, return_tensors="pt", padding=True, truncation=True,max_length=500, add_special_tokens = True)

        pos_output = model.forward(**pos_encoded).squeeze(1)
        neg_output = model.forward(**neg_encoded).squeeze(1)

        labels = torch.zeros(1, dtype=torch.long)

        loss = cross_entropy(torch.stack((pos_output, neg_output), dim=1), labels)

        loss.backward()
        optimizer.step()
    j += 1

filename = 'model.sav'
pickle.dump(model, open(filename, 'wb'))