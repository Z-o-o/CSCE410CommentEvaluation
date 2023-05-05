from torch.optim import Adam
from tqdm import tqdm
from torch import nn
from torch import nn
from evaluate import evaluate
import torch
import numpy as np
import pandas as pd
import custom_dataset
import get_qa_threads
import model


def train(custom_model, train_data, val_data, learning_rate, epochs):
    train, val = custom_dataset.CustomDataset(train_data), custom_dataset.CustomDataset(
        val_data
    )

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=2, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=2)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(custom_model.parameters(), lr=learning_rate)

    if use_cuda:
        custom_model = custom_model.cuda()
        criterion = criterion.cuda()

    for epoch_num in range(epochs):
        total_acc_train = 0
        total_loss_train = 0

        for train_input, train_label in tqdm(train_dataloader):
            train_label = train_label.to(device)
            mask = train_input["attention_mask"].to(device)
            input_id = train_input["input_ids"].squeeze(1).to(device)

            output = custom_model(input_id, mask)

            batch_loss = criterion(output, train_label.long())
            total_loss_train += batch_loss.item()

            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc

            custom_model.zero_grad()
            batch_loss.backward()
            optimizer.step()

        total_acc_val = 0
        total_loss_val = 0

        with torch.no_grad():
            for val_input, val_label in val_dataloader:
                val_label = val_label.to(device)
                mask = val_input["attention_mask"].to(device)
                input_id = val_input["input_ids"].squeeze(1).to(device)

                output = custom_model(input_id, mask)

                batch_loss = criterion(output, val_label.long())
                total_loss_val += batch_loss.item()

                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc

        print(
            f"Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} \
                | Train Accuracy: {total_acc_train / len(train_data): .3f} \
                | Val Loss: {total_loss_val / len(val_data): .3f} \
                | Val Accuracy: {total_acc_val / len(val_data): .3f}"
        )

        print("Saving model to first-model.pth")
        torch.save(custom_model.state_dict(), "first-model.pth")


def evaluate(model, test_data):
    test = custom_dataset.CustomDataset(test_data)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=2)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()

    total_acc_test = 0
    with torch.no_grad():
        for test_input, test_label in test_dataloader:
            test_label = test_label.to(device)
            mask = test_input["attention_mask"].to(device)
            input_id = test_input["input_ids"].squeeze(1).to(device)

            output = model(input_id, mask)

            acc = (output.argmax(dim=1) == test_label).sum().item()
            total_acc_test += acc

    print(f"Test Accuracy: {total_acc_test / len(test_data): .3f}")

    # model.eval()
    # predict(model, text='Christiano Ronaldo scored 2 goals in last Manchester United game')


if __name__ == "__main__":
    # TODO: combine all datasets and split them properly
    py_data = get_qa_threads.itemize_data("data/python_database.pkl")
    py_df = get_qa_threads.convert_data_to_df(py_data)
    math_data = get_qa_threads.itemize_data("data/math_database.pkl")
    math_df = get_qa_threads.convert_data_to_df(math_data)
    design_data = get_qa_threads.itemize_data("data/design_database.pkl")
    design_df = get_qa_threads.convert_data_to_df(design_data)
    # df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42), [int(100), int(200)])
    py_train, py_val, py_test = np.split(
        py_df.sample(frac=1, random_state=42),
        [int(0.8 * len(py_df)), int(0.9 * len(py_df))],
    )
    math_train, math_val, math_test = np.split(
        math_df.sample(frac=1, random_state=42),
        [int(0.8 * len(math_df)), int(0.9 * len(math_df))],
    )
    design_train, design_val, design_test = np.split(
        design_df.sample(frac=1, random_state=42),
        [int(0.8 * len(design_df)), int(0.9 * len(design_df))],
    )
    custom_model = model.BertClassifier()
    df_train = pd.concat([py_train, math_train, design_train])
    df_val = pd.concat([py_val, math_val, design_val])
    df_test = pd.concat([py_test, math_test, design_test])
    print("Starting Training\n")
    train(custom_model, df_train, df_val, 1e-6, 11)
    print("Starting Evaluation")
    evaluate(custom_model, df_test)
