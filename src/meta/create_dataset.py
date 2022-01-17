import os
import random

import pandas as pd
from datasets import Dataset, DatasetDict, load_from_disk


def main():
    dataset = load_from_disk("../../data/muvuldeepecker/full")

    train_df = dataset["train"].to_pandas()
    val_test_df = dataset["val_test"].to_pandas()
    dataset_df = pd.concat([train_df, val_test_df])

    labels = dataset_df["vul_label"].unique().tolist()
    labels.remove(0)
    random.shuffle(labels)
    labels = [0] + labels

    train_labels = labels[:33]
    val_labels = labels[33:37]
    test_labels = labels[37:]

    train_df_list = []
    val_df_list = []
    test_df_list = []
    val_test_df_list = []
    for label in labels:
        print(f"precessing {label} label")
        df = dataset_df[dataset_df["vul_label"] == label]
        if label in train_labels:
            train_df_list.append(df)
        else:
            val_test_df_list.append(df)
            if label in val_labels:
                val_df_list.append(df)
            elif label in test_labels:
                test_df_list.append(df)
    train_df = pd.concat(train_df_list)
    val_df = pd.concat(val_df_list)
    test_df = pd.concat(test_df_list)
    val_test_df = pd.concat(val_test_df_list)

    save_path = "../../data/muvuldeepecker/meta"
    train_set = Dataset.from_pandas(train_df)
    val_set = Dataset.from_pandas(val_df)
    test_set = Dataset.from_pandas(test_df)
    val_test_set = Dataset.from_pandas(val_test_df)

    dataset_dict = DatasetDict()
    dataset_dict["train"] = train_set
    dataset_dict["validation"] = val_set
    dataset_dict["test"] = test_set
    dataset_dict["val_test"] = val_test_set
    dataset_dict = dataset_dict.remove_columns(["__index_level_0__"])
    dataset_dict.save_to_disk(save_path)
    print(f"train: {len(train_set)}, val_test: {len(val_test_df)}")


def create_meta_sysevr():
    dataset_path = "../../data/sysevr"
    train = ["API_function_call", "Arithmetic_expression", "Pointer_usage"]
    test = ["Array_usage"]

    train_df_list = []
    for name in train:
        print(f"processing {name}")
        path = os.path.join(dataset_path, name)
        dataset = load_from_disk(path)
        train_df = dataset["train"].to_pandas()
        val_test_df = dataset["val_test"].to_pandas()
        dataset_df = pd.concat([train_df, val_test_df])
        train_df_list.append(dataset_df)
    train_df = pd.concat(train_df_list)
    train_set = Dataset.from_pandas(train_df)

    test_df_list = []
    for name in test:
        print(f"processing {name}")
        path = os.path.join(dataset_path, name)
        dataset = load_from_disk(path)
        train_df = dataset["train"].to_pandas()
        val_test_df = dataset["val_test"].to_pandas()
        dataset_df = pd.concat([train_df, val_test_df])
        test_df_list.append(dataset_df)
    test_df = pd.concat(test_df_list)
    test_set = Dataset.from_pandas(test_df)

    dataset = DatasetDict()
    dataset["train"] = train_set
    dataset["test"] = test_set
    dataset = dataset.remove_columns(["__index_level_0__"])

    save_path = os.path.join(dataset_path, "meta")
    os.makedirs(save_path, exist_ok=True)
    dataset.save_to_disk(save_path)


if __name__ == "__main__":
    # main()
    create_meta_sysevr()
