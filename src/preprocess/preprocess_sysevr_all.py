import logging
import os

import pandas as pd
from codegen.cpp_processor import CppProcessor
from datasets import Dataset, DatasetDict, load_from_disk
from normalizers import remove_comments, remove_empty_lines, remove_space_before_newline
from tokenize_utils import tokenize_fn

log = logging.getLogger(__name__)


def get_sysevr(data_dir: str):
    vul_files = ["API_function_call", "Arithmetic_expression", "Array_usage", "Pointer_usage"]
    save_path = os.path.join(data_dir, "sysevr", "raw_data")
    if not os.path.exists(os.path.join(save_path, "train")):
        for file in vul_files:
            file_path = os.path.join(save_path, f"{file}.txt")
            all_examples = []
            with open(file_path) as fp:
                v, nv = 0, 0
                lines = fp.readlines()
                example = []
                for idx, line in enumerate(lines):
                    if line.strip() == "":
                        continue
                    if "-------------------------" in line:
                        if len(example) >= 3:
                            code = "\n".join(example[1:-1])
                            try:
                                label = int(example[-1])
                                if label == 0:
                                    nv += 1
                                else:
                                    v += 1
                                all_examples.append(
                                    {
                                        "code": code,
                                        "label": label,
                                    }
                                )
                            except ValueError:
                                pass
                            example = []
                    else:
                        example.append(line.strip())
                log.info(v, nv)
        df = pd.DataFrame(all_examples)
        dataset = Dataset.from_pandas(df)
        train, test = dataset.train_test_split(test_size=0.2).values()
        val, test = test.train_test_split(test_size=0.5).values()
        dataset = DatasetDict()
        dataset["train"] = train
        dataset["validation"] = val
        dataset["test"] = test
        dataset.save_to_disk(save_path)
    else:
        dataset = load_from_disk(save_path)
    return dataset


def tokenize_sysevr(data_dir: str):
    save_path = os.path.join(data_dir, "sysevr", "tokenized")
    if not os.path.exists(os.path.join(save_path, "train")):
        os.makedirs(save_path, exist_ok=True)
        dataset = get_sysevr(data_dir)
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
    tokenize_sysevr(os.path.join(root_path, "data"))


if __name__ == "__main__":
    main()
