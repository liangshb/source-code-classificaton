import os

import pandas as pd
from datasets import Dataset

from src.preprocess.deepwukong_symbolizer import tokenize_lines
from src.preprocess.pipeline import split_pipeline
from src.preprocess.utils import merge_lines


def read_file(file_path: str, vul_label: int, vul_str: str):
    df = pd.read_json(file_path)
    df = df.rename(columns={"target": "label"})
    df["vul_label"] = df["label"]
    df.loc[df["vul_label"] == 1, "vul_label"] = vul_label
    df["vul_str"] = vul_str
    nv = df.loc[df["label"] == 0, "label"].count().item()
    v = df.loc[df["label"] == 1, "label"].count().item()
    print("Positive: {}, Negative: {}".format(v, nv))
    return df


def deepwukong_pipeline(dataset: Dataset):
    print("=================tokens-sym-no===================")
    print("tokenize symbolized lines")
    dataset = dataset.map(
        lambda example: tokenize_lines(example, "nodes-line-sym", "sym-no"),
        batched=True,
    )
    print("merge lines")
    dataset = dataset.map(
        lambda example: merge_lines(example, "tokens-sym-no", "tokens-sym-no"),
        batched=True,
    )
    return dataset


def preprocess_deepwukong(data_dir: str, dataset_name: str = "deepwukong"):
    dataset_path = os.path.join(data_dir, dataset_name)
    raw_data_path = os.path.join(dataset_path, "raw_data")
    vul_files = [file for file in os.listdir(raw_data_path) if file.endswith(".json")]

    for vul_label, file in enumerate(vul_files, start=1):
        print(f"Processing {file}")
        file_path = os.path.join(raw_data_path, file)
        vul_str = file.split(".")[0]
        vul_df = read_file(file_path, vul_label, vul_str)
        save_path = os.path.join(dataset_path, vul_str)
        if not os.path.exists(os.path.join(save_path, "dataset_dict.json")):
            os.makedirs(save_path, exist_ok=True)
            dataset = Dataset.from_pandas(vul_df)
            dataset = deepwukong_pipeline(dataset)
            dataset_dict = split_pipeline(dataset)
            dataset_dict.save_to_disk(save_path)
            print(f"{vul_str} dataset dict saved")


def main():
    data_dir = "../../data"
    dataset_name = ["deepwukong"]
    for name in dataset_name:
        preprocess_deepwukong(data_dir, name)


if __name__ == "__main__":
    main()
