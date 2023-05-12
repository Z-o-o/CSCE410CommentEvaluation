import torch
import custom_dataset
import pandas as pd
import numpy as np
import sklearn.metrics
from matplotlib import pyplot as plt

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
    y_true = []
    y_pred = []
    model.eval()
    with open(filename, "w", encoding="utf-8") as f:
        f.write("text,ground_truth,prediction\n")
        with torch.no_grad():
            i = 0
            for test_input, test_label in test_dataloader:
                test_label = test_label.to(device)
                mask = test_input["attention_mask"].to(device)
                input_id = test_input["input_ids"].squeeze(1).to(device)

                outputs = model(input_id, mask)
                preds = torch.argmax(outputs.data, dim=1)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                conf, _ = torch.max(probs, 1)
                y_true = np.append(y_true, torch.max(test_label.cpu()))
                y_pred = np.append(y_pred, conf.max().item())
                CM += sklearn.metrics.confusion_matrix(
                    test_label.cpu(), preds.cpu(), labels=[0, 1]
                )
                ground_truth = torch.max(test_label)
                prediction = torch.max(preds)
                text = test_data.iloc[[i]]["text"].values[0]
                text = text.strip().replace(",", "").replace("\n", "")
                f.write(f"{text},{ground_truth},{prediction}\n")
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
        print()
        return y_true, y_pred


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
    custom_model.load_state_dict(torch.load("first-model.pth"))
    # df_train = pd.concat([py_train, math_train, design_train])
    # df_val = pd.concat([py_val, math_val, design_val])
    df_test = pd.concat([py_test, math_test, design_test])
    print("Starting Test Evaluation")
    trained_true, trained_pred = evaluate(
        custom_model, df_test, "trained_test_output.csv"
    )
    del custom_model
    custom_model = model.BertClassifier()
    untrained_true, untrained_pred = evaluate(
        custom_model, df_test, "untrained_test_output.csv"
    )

    # RocCurveDisplay.from_predictions(y_true, y_pred, pos_label=0)
    fpr, tpr, _ = sklearn.metrics.roc_curve(trained_true, trained_pred, pos_label=0)
    auc = round(sklearn.metrics.roc_auc_score(trained_true, trained_pred), 4)
    plt.plot(fpr, tpr, label="Trained, AUC=" + str(auc))

    fpr, tpr, _ = sklearn.metrics.roc_curve(untrained_true, untrained_pred, pos_label=0)
    auc = round(sklearn.metrics.roc_auc_score(untrained_true, untrained_pred), 4)
    plt.plot(fpr, tpr, label="Untrained, AUC=" + str(auc))
    plt.legend()

    plt.savefig("roc_curve.png")
