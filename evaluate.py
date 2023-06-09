import time
import torch
import custom_dataset
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

import get_qa_threads
import custom_dataset
import model


def evaluate(model, test_data, filename):
    test = custom_dataset.CustomDataset(test_data)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=2, shuffle=False)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()



    CM = 0
    model.eval()
    with open(f'{filename}-1.csv', "w", encoding="utf-8") as f:  
        f.write("ground_truth,prediction,probability\n")
        with torch.no_grad():
            i = 0
            for test_input, test_label in test_dataloader:
                test_label = test_label.to(device)
                mask = test_input["attention_mask"].to(device)
                input_id = test_input["input_ids"].squeeze(1).to(device)

                outputs = model(input_id, mask)
                probs = torch.nn.functional.softmax(outputs.data, dim=1)
                preds = torch.argmax(outputs.data, dim=1)
                CM += confusion_matrix(test_label.cpu(), preds.cpu(), labels=[0, 1])
                # ground_truth = torch.max(test_label)
                # prediction = torch.max(preds)
                # probability = torch.max(probs)
                # text = test_data.iloc[[i]]["text"].values[0]
                # text = text.strip().replace(",", "").replace("\n", "")
                # f.write(f"{text},{ground_truth},{prediction},{probability}\n")
                for j in range(0, len(preds.cpu())):
                    ground_truth = test_label[j]
                    prediction = preds[j]
                    probability = probs[j].cpu().numpy()[0]
                    f.write(f"{ground_truth},{prediction},{probability}\n")
                i += 1

        tn = CM[1][1]
        tp = CM[0][0]
        fp = CM[1][0]
        fn = CM[0][1]
        acc = np.sum(np.diag(CM) / np.sum(CM))
        sensitivity = tp / (tp + fn)
        precision = tp / (tp + fp)

        print("\nTestset Accuracy(mean): %f %%" % (100 * acc))
        print()
        print("Confusion Matirx : ")
        print(CM)
        print("- Sensitivity : ", (tp / (tp + fn)) * 100)
        print("- Specificity : ", (tn / (tn + fp)) * 100)
        print("- Precision: ", (tp / (tp + fp)) * 100)
        print("- NPV: ", (tn / (tn + fn)) * 100)
        print(
            "- F1 : ", ((2 * sensitivity * precision) / (sensitivity + precision)) * 100
        )
        # out_df.to_csv(filename, index=False)
        print(f'Final i={i}\n')


if __name__ == "__main__":
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
    custom_model.load_state_dict(torch.load("test-model.pth"))
    # df_train = pd.concat([py_train, math_train, design_train])
    df_val = pd.concat([py_val, math_val, design_val])
    df_test = pd.concat([py_test, math_test, design_test])
    start_time = time.time()
    print("Starting Test Evaluation")
    evaluate(custom_model, df_test, "test_output")
    end_time = time.time()
    print(f"Runtime: {(end_time - start_time)}s\n")

    start_time = time.time()
    print("Starting Val Evaluation")
    evaluate(custom_model, df_val, "val_output")
    end_time = time.time()
    print(f"Runtime: {(end_time - start_time)}s\n")
