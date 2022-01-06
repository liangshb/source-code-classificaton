import logging
import os

import pandas as pd
from codegen.cpp_processor import CppProcessor
from datasets import Dataset, DatasetDict, load_from_disk
from normalizers import remove_comments, remove_empty_lines, remove_space_before_newline

from src.preprocess.tokenizers.tokenize_utils import tokenize_fn

log = logging.getLogger(__name__)


def get_reveal(data_dir: str):
    save_path = os.path.join(data_dir, "reveal", "raw_data")
    if not os.path.exists(os.path.join(save_path, "train")):
        df_vul = pd.read_json(os.path.join(save_path, "vulnerables.json"))
        df_non_vul = pd.read_json(os.path.join(save_path, "non-vulnerables.json"))
        df_vul["label"] = 1
        df_non_vul["label"] = 0
        df = pd.concat([df_vul, df_non_vul])
        dataset = Dataset.from_pandas(df)
        train, test = dataset.train_test_split(test_size=0.2).values()
        val, test = test.train_test_split(test_size=0.5).values()
        dataset = DatasetDict()
        dataset["train"] = train
        dataset["validation"] = val
        dataset["test"] = test
        dataset = dataset.remove_columns(["__index_level_0__", "hash", "size", "project"])
        dataset.save_to_disk(save_path)
    else:
        dataset = load_from_disk(save_path)
    return dataset


def tokenize_reveal(data_dir: str):
    save_path = os.path.join(data_dir, "reveal", "tokenized")
    if not os.path.exists(os.path.join(save_path, "train")):
        os.makedirs(save_path, exist_ok=True)
        dataset = get_reveal(data_dir)
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
            lambda example: tokenize_fn(encode_fn, example, key="code"), batched=False
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
    tokenize_reveal(os.path.join(root_path, "data"))


if __name__ == "__main__":
    main()
