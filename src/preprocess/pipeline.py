from datasets import DatasetDict

from src.preprocess.normalizers import normalize_fn
from src.preprocess.symbolizer import symbolize_str_char, symbolize_var_func_fn
from src.preprocess.tokenizer import cpp_tokenizer, java_tokenizer


def preprocess_pipeline(
    dataset: DatasetDict, code_key: str = "code", language: str = "cpp"
):
    # normalize
    dataset = dataset.map(lambda example: normalize_fn(example, code_key), batched=True)

    # symbolize phase1: symbolize variable and function, add code-sym
    dataset = dataset.map(lambda example: symbolize_var_func_fn(example, key=code_key))

    # tokenize: add tokens, tags
    if language == "cpp":
        dataset = dataset.map(
            lambda example: cpp_tokenizer(example, code_key), batched=True
        )
        dataset = dataset.map(
            lambda example: cpp_tokenizer(example, f"{code_key}-sym", postfix="-sym"),
            batched=True,
        )
    elif language == "java":
        dataset = dataset.map(
            lambda example: java_tokenizer(example, code_key), batched=True
        )
        dataset = dataset.map(
            lambda example: java_tokenizer(example, f"{code_key}-sym", postfix="-sym"),
            batched=True,
        )
    else:
        print("Not supported language")
        exit(-1)

    # symbolize phase2: symbolize string and char
    dataset = dataset.map(
        lambda example: symbolize_str_char(example, "tokens"), batched=True
    )
    dataset = dataset.map(
        lambda example: symbolize_str_char(example, "tokens-sym"), batched=True
    )

    return dataset
