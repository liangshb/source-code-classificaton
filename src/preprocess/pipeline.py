from datasets import Dataset, DatasetDict
from datasets.arrow_dataset import Batch

from src.preprocess.normalizers import normalize_fn
from src.preprocess.symbolizer import (
    symbolize_code_str_char_hash,
    symbolize_code_str_char_len,
    symbolize_code_var_func,
)
from src.preprocess.tokenizer import cpp_tokenizer, java_tokenizer


def preprocess_pipeline(dataset: Dataset, code_key: str = "code", language: str = "cpp"):
    # normalize
    print("normalizing")
    dataset = dataset.map(lambda example: normalize_fn(example, code_key), batched=True)

    # symbolize phase1: symbolize string and char, add code-hash, code-len
    print("symbolizing phase1")
    dataset = dataset.map(
        lambda example: symbolize_code_str_char_hash(example, code_key=code_key), batched=True
    )
    dataset = dataset.map(
        lambda example: symbolize_code_str_char_len(example, code_key=code_key), batched=True
    )

    # symbolize phase2: symbolize func and var
    print("symbolizing phase2")
    dataset = dataset.map(
        lambda example: symbolize_code_var_func(example, code_key="code-hash"), batched=True
    )
    dataset = dataset.map(
        lambda example: symbolize_code_var_func(example, code_key="code-len"), batched=True
    )

    # tokenize: add tokens-hash, tokens-len, tags-hash, tags-len
    print("tokenizing")
    if language == "cpp":
        dataset = dataset.map(
            lambda example: cpp_tokenizer(example, key="code-hash", postfix="-hash"),
            batched=True,
        )
        dataset = dataset.map(
            lambda example: cpp_tokenizer(example, key="code-len", postfix="-len"),
            batched=True,
        )
    elif language == "java":
        dataset = dataset.map(
            lambda example: java_tokenizer(example, key="code-hash", postfix="-hash"),
            batched=True,
        )
        dataset = dataset.map(
            lambda example: java_tokenizer(example, key="code-len", postfix="-len"),
            batched=True,
        )
    else:
        print("Not supported language")
        exit(-1)

    return dataset


def split_pipeline(dataset: Dataset) -> DatasetDict:
    train, val_test = dataset.train_test_split(test_size=0.2).values()
    val, test = val_test.train_test_split(test_size=0.5).values()
    dataset_dict = DatasetDict()
    dataset_dict["train"] = train
    dataset_dict["validation"] = val
    dataset_dict["test"] = test
    dataset_dict["val_test"] = val_test
    return dataset_dict
