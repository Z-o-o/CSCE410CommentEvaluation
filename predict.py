from transformers import BertModel
from transformers import BertTokenizer
import custom_dataset
import torch


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
        label_id = output.argmax(dim=1).item()
        for key in custom_dataset.labels.keys():
            if custom_dataset.labels[key] == label_id:
                print(text, " => ", key, "#", label_id)
                break
