from datasets import Dataset, DatasetDict

from src.preprocess.normalizers import normalize_fn
from src.preprocess.symbolizers.symbolizer import (
    reval_symbolize,
    symbolize_code_str_char_hash,
    symbolize_code_str_char_len,
    symbolize_code_var_func,
    symbolize_identifier,
    symbolize_str_char_hash,
    symbolize_str_char_len,
)
from src.preprocess.tokenizers.tree_sitter_tokenizer import (
    cpp_tokenizer,
    java_tokenizer,
    nltk_tokenizer,
)


def reveal_pipeline(dataset: Dataset, code_key: str = "code"):
    print("normalizing")
    dataset = dataset.map(lambda example: normalize_fn(example, code_key), batched=True)

    print("tokenize")
    dataset = dataset.map(lambda example: nltk_tokenizer(example, code_key), batched=True)

    print("symbolize")
    dataset = dataset.map(lambda example: reval_symbolize(example), batched=True)

    return dataset


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


def pipeline_tokenize_first(dataset: Dataset, code_key: str = "code", language: str = "cpp"):
    # normalize
    print("normalizing")
    dataset = dataset.map(lambda example: normalize_fn(example, code_key), batched=True)

    # tokenize: add tokens, tags
    print("tokenizing")
    if language == "cpp":
        dataset = dataset.map(
            lambda example: cpp_tokenizer(example, key=code_key),
            batched=True,
        )
    elif language == "java":
        dataset = dataset.map(
            lambda example: java_tokenizer(example, key=code_key),
            batched=True,
        )
    else:
        print("Not supported language")
        exit(-1)

    # symbolize identifier: add tokens-sym
    print("symbolize identifier")
    dataset = dataset.map(lambda example: symbolize_identifier(example), batched=True)
    # symbolize str and char
    print("symbolize string and char")
    dataset = dataset.map(lambda example: symbolize_str_char_hash(example), batched=True)
    dataset = dataset.map(lambda example: symbolize_str_char_len(example), batched=True)

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
