import os

import pandas as pd
from datasets import Dataset

from src.preprocess.pipeline import preprocess_pipeline, split_pipeline


def read_file(file_path: str, vul_label: int, vul_str: str):
    all_examples = []
    with open(file_path) as fp:
        v, nv = 0, 0
        lines = fp.readlines()
        example = []
        for line in lines:
            if line.strip() == "":
                continue
            if "-------------------------" in line:
                if len(example) >= 3:
                    code = "\n".join(example[1:-1])
                    try:
                        label = int(example[-1])
                        if label == 0:
                            nv += 1
                        elif label == 1:
                            v += 1
                        else:
                            print(label)
                            continue
                        all_examples.append(
                            {
                                "code": code,
                                "label": label,
                                "vul_label": label and vul_label,
                                "vul_str": vul_str,
                            }
                        )
                    except ValueError:
                        pass
                    example = []
            else:
                example.append(line.strip())
        print(f"Positive: {v}, Negative: {nv}")
    return all_examples


def read_file_mu(file_path: str):
    all_examples = []
    with open(file_path) as fp:
        v, nv = 0, 0
        lines = fp.readlines()
        example = []
        for line in lines:
            if line.strip() == "":
                continue
            if "-------------------------" in line:
                if len(example) >= 3:
                    code = "\n".join([" ".join(e.split()[:-1]) for e in example[1:-1]])
                    try:
                        label = int(example[-1])
                        if label == 0:
                            nv += 1
                        else:
                            v += 1
                        all_examples.append(
                            {
                                "code": code,
                                "label": int(label != 0),
                                "vul_label": label,
                            }
                        )
                    except ValueError:
                        pass
                    example = []
            else:
                example.append(line.strip())
        print(f"Positive: {v}, Negative: {nv}")
    return all_examples


def preprocess_sysevr(data_dir: str, dataset_name: str = "sysevr"):
    dataset_path = os.path.join(data_dir, dataset_name)
    raw_data_path = os.path.join(dataset_path, "raw_data")
    vul_files = [file for file in os.listdir(raw_data_path) if file.endswith(".txt")]

    full_examples = []
    for vul_label, file in enumerate(vul_files):
        print(f"Processing {file}")
        file_path = os.path.join(raw_data_path, file)
        vul_str = file.split(".")[0]
        vul_examples = read_file(file_path, vul_label, vul_str)
        save_path = os.path.join(dataset_path, vul_str)
        if not os.path.exists(os.path.join(save_path, "dataset_dict.json")):
            os.makedirs(save_path, exist_ok=True)
            dataset = Dataset.from_pandas(pd.DataFrame(vul_examples))
            dataset = preprocess_pipeline(dataset)
            dataset_dict = split_pipeline(dataset)
            dataset_dict.save_to_disk(save_path)
            print(f"{vul_str} dataset dict saved")
        full_examples += vul_examples

    # full dataset
    save_path = os.path.join(dataset_path, "full")
    if not os.path.exists(os.path.join(save_path, "dataset_dict.json")):
        os.makedirs(save_path, exist_ok=True)
        dataset = Dataset.from_pandas(pd.DataFrame(full_examples))
        dataset = preprocess_pipeline(dataset)
        dataset_dict = split_pipeline(dataset)
        dataset_dict.save_to_disk(save_path)
        print("full dataset dict saved")


def preprocess_muvuldeepecker(data_dir: str, dataset_name: str = "muvuldeepecker"):
    dataset_path = os.path.join(data_dir, dataset_name)
    raw_data_path = os.path.join(dataset_path, "raw_data")
    save_path = os.path.join(dataset_path, "full")
    vul_file = os.path.join(raw_data_path, "mvd.txt")
    if not os.path.exists(os.path.join(save_path, "dataset_dict.json")):
        os.makedirs(save_path, exist_ok=True)
        vul_examples = read_file_mu(vul_file)
        dataset = Dataset.from_pandas(pd.DataFrame(vul_examples))
        dataset = preprocess_pipeline(dataset)
        dataset_dict = split_pipeline(dataset)
        dataset_dict.save_to_disk(save_path)
        print("full dataset dict saved")


def main():
    data_dir = "../../data"
    dataset_name = ["sysevr", "vuldeepecker", "muvuldeepecker"]
    for name in dataset_name:
        print(f"======================Processing {name}==========================")
        if name == "muvuldeepecker":
            preprocess_muvuldeepecker(data_dir)
        else:
            preprocess_sysevr(data_dir, name)


if __name__ == "__main__":
    main()
