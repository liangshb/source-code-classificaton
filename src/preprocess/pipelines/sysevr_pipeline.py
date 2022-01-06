from datasets import Dataset

from src.preprocess.normalizers import normalize_list_fn
from src.preprocess.symbolizers.sysevr_symbolizer import symbolize_fn
from src.preprocess.tokenizers.sysevr_tokenizer import tokenize_fn
from src.preprocess.utils import merge_lines_tokens


def sysevr_pipeline(dataset: Dataset, code_key: str = "code"):
    print("normalizing")
    dataset = dataset.map(lambda example: normalize_list_fn(example, code_key), batched=True)

    print("tokenizing")
    dataset = dataset.map(lambda example: tokenize_fn(example, code_key), batched=True)

    print("symbolizing")
    dataset = dataset.map(lambda example: symbolize_fn(example), batched=True)

    print("tokenizing symbolized code")
    dataset = dataset.map(
        lambda example: tokenize_fn(example, "code-sym", tokens_key="tokens-sym"), batched=True
    )

    print("merge lines")
    dataset = dataset.map(lambda example: merge_lines_tokens(example), batched=True)

    return dataset
