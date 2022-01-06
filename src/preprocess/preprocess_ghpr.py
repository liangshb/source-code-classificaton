import logging
import os

import pandas as pd
from codegen.java_processor import JavaProcessor
from datasets import Dataset, DatasetDict, load_from_disk
from normalizers import remove_comments, remove_empty_lines, remove_space_before_newline
from pandas import DataFrame

from src.preprocess.tokenizers.tokenize_utils import tokenize_fn

log = logging.getLogger(__name__)


def get_ghpr(data_dir: str):
    save_path = os.path.join(data_dir, "ghpr", "raw_data")
    if not os.path.exists(os.path.join(save_path, "train")):
        names = ["language", "old_code", "new_code"]
        df: DataFrame = pd.read_csv(
            os.path.join(save_path, "ghprdata.csv"),
            names=names,
            usecols=[4, 10, 11],
        )
        df = df[df["language"] == "Java"]
        vul_series = df.pop("old_code")
        non_vul_series = df.pop("new_code")
        vul_df = pd.DataFrame(
            {
                "code": vul_series,
                "label": 1,
            }
        )
        non_vul_df = pd.DataFrame({"code": non_vul_series, "label": 0})
        df = pd.concat([vul_df, non_vul_df])
        dataset = Dataset.from_pandas(df)
        train, test = dataset.train_test_split(test_size=0.2).values()
        val, test = test.train_test_split(test_size=0.5).values()
        dataset = DatasetDict()
        dataset["train"] = train
        dataset["validation"] = val
        dataset["test"] = test
        dataset = dataset.remove_columns(["__index_level_0__"])
        dataset.save_to_disk(save_path)
    else:
        dataset = load_from_disk(save_path)
    return dataset


def tokenize_ghpr(data_dir: str):
    save_path = os.path.join(data_dir, "ghpr", "tokenized")
    if not os.path.exists(os.path.join(save_path, "train")):
        os.makedirs(save_path, exist_ok=True)
        dataset = get_ghpr(data_dir)
        log.info("Processing dataset")

        # normalize
        dataset = dataset.map(lambda example: {"code": remove_comments(example["code"])})
        dataset = dataset.map(
            lambda example: {"code": remove_space_before_newline(example["code"])}
        )
        dataset = dataset.map(lambda example: {"code": remove_empty_lines(example["code"])})

        # tokenize
        encode_fn = JavaProcessor(
            root_folder=os.path.join(data_dir, "..", "src", "preprocess"),
        )
        dataset = dataset.map(
            lambda example: tokenize_fn(encode_fn, example, key="code"), batched=True
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
    tokenize_ghpr(os.path.join(root_path, "data"))


if __name__ == "__main__":
    main()
