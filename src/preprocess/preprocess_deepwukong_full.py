import logging
import os

import pandas as pd
from codegen.cpp_processor import CppProcessor
from datasets import Dataset, DatasetDict, load_from_disk
from normalizers import remove_comments, remove_empty_lines, remove_space_before_newline
from tokenize_utils import tokenize_fn

log = logging.getLogger(__name__)


def merge_lines(example, key: str = "nodes-line", out_key: str = "code"):
    code = [" ".join(lines).strip() for lines in example[key]]
    return {out_key: code}


def get_deepwukong(data_dir: str):
    save_path = os.path.join(data_dir, "deepwukong", "raw_data")
    if not os.path.exists(os.path.join(save_path, "dataset_dict.json")):
        file_list = [d for d in os.listdir(save_path) if d.endswith(".json")]
        df_list = []
        for file in file_list:
            log.info(f"Processing {file}")
            cwe = file.split(".")[0]
            df = pd.read_json(os.path.join(save_path, file))
            df = df.drop(columns=["filePathList", "testcaseID"])
            df["cwe"] = cwe
            df_list.append(df)
        df = pd.concat(df_list)
        dataset = Dataset.from_pandas(df)
        train, test = dataset.train_test_split(test_size=0.2).values()
        val, test = test.train_test_split(test_size=0.5).values()
        dataset = DatasetDict()
        dataset["train"] = train
        dataset["validation"] = val
        dataset["test"] = test
        dataset = dataset.remove_columns(["__index_level_0__"])
        dataset.rename_column("target", "label")
        dataset = dataset.map(lambda example: merge_lines(example), batched=True)
        dataset = dataset.map(
            lambda example: merge_lines(example, key="nodes-line-sym", out_key="code-sym"),
            batched=True,
        )
        dataset.save_to_disk(save_path)
    else:
        dataset = load_from_disk(save_path)
    return dataset


def tokenize_deepwukong(data_dir: str):
    save_path = os.path.join(data_dir, "deepwukong", "tokenized")
    if not os.path.exists(os.path.join(save_path, "dataset_dict.json")):
        os.makedirs(save_path, exist_ok=True)
        dataset = get_deepwukong(data_dir)
        log.info("Processing dataset")

        # normalize
        dataset = dataset.map(lambda example: {"code": remove_comments(example["code"])})
        dataset = dataset.map(
            lambda example: {"code": remove_space_before_newline(example["code"])}
        )
        dataset = dataset.map(lambda example: {"code": remove_empty_lines(example["code"])})

        # tokenize
        encode_fn = CppProcessor(
            root_folder=os.path.join(data_dir, "..", "src", "preprocess"),
        )
        dataset = dataset.map(
            lambda example: tokenize_fn(encode_fn, example, key="code"), batched=True
        )
        dataset = dataset.map(
            lambda example: tokenize_fn(encode_fn, example, key="code-sym", postfix="-sym"),
            batched=True,
        )
        os.makedirs(save_path, exist_ok=True)
        log.info("Processing done")

        dataset.save_to_disk(save_path)
        log.info("Saving done")
    else:
        log.info("Loading from previous saved")
        _ = load_from_disk(save_path)
    pass


def main():
    root_path = "../../"
    print(os.path.abspath(os.path.join(root_path, "data")))
    tokenize_deepwukong(os.path.join(root_path, "data"))


if __name__ == "__main__":
    main()
