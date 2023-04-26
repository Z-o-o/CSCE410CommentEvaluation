from transformers import BertTokenizer
import torch
import argparse
import custom_dataset
import model
import get_qa_threads


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
                print(text, " => ", key, "#", label_id)
                print(conf)
                print(conf.item())
                break


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", "-t")


if __name__ == "__main__":
    custom_model = model.BertClassifier()
    custom_model.load_state_dict(torch.load("test-model.pth"))
    predict(
        custom_model,
        """Does Chunders like Sake?""",
    )
