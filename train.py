from torch.optim import Adam
from tqdm import tqdm
from torch import nn
from torch import nn
from evaluate import evaluate
import torch
import numpy as np
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


if __name__ == "__main__":
    # TODO: combine all datasets and split them properly
    py_data = get_qa_threads.itemize_data("data/python_database.pkl")
    df = get_qa_threads.convert_data_to_df(py_data)
    df_train, df_val, df_test = np.split(
        df.sample(frac=1, random_state=42), [int(100), int(200)]
    )
    custom_model = model.BertClassifier()
    train(custom_model, df_train, df_val, 1e-6, 3)
    evaluate(custom_model, df_test)
    # model.eval()
    # predict(model, text='Christiano Ronaldo scored 2 goals in last Manchester United game')
