from datasets import Dataset, DatasetDict
from datasets.arrow_dataset import Batch

from src.preprocess.normalizers import normalize_fn
from src.preprocess.symbolizer import symbolize_identifier, symbolize_str_char, symbolize_str_char2
from src.preprocess.tokenizer import cpp_tokenizer, java_tokenizer


def preprocess_pipeline(dataset: Dataset, code_key: str = "code", language: str = "cpp"):
    # normalize
    print("normalizing")
    dataset = dataset.map(lambda example: normalize_fn(example, code_key), batched=True)

    # tokenize: add tokens, tags
    print("tokenizing")
    if language == "cpp":
        dataset = dataset.map(lambda example: cpp_tokenizer(example, code_key), batched=True)
    elif language == "java":
        dataset = dataset.map(lambda example: java_tokenizer(example, code_key), batched=True)
    else:
        print("Not supported language")
        exit(-1)

    # symbolize phase1: symbolize identifier: add tokens-sym
    print("symbolizing phase1")
    dataset = dataset.map(
        lambda example: symbolize_identifier(example, tokens_key="tokens", tags_key="tags"),
        batched=True,
    )

    # symbolize phase2: symbolize string and char
    print("symbolizing phase2")
    dataset = dataset.map(
        lambda example: symbolize_str_char(example, tokens_key="tokens", tags_key="tags"),
        batched=True,
    )
    dataset = dataset.map(
        lambda example: symbolize_str_char2(example, tokens_key="tokens-sym", tags_key="tags"),
        batched=True,
    )

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
