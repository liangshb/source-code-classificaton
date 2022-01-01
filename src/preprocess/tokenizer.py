from src.preprocess.codegen.cpp_processor import CppProcessor
from src.preprocess.codegen.java_processor import JavaProcessor


def cpp_tokenizer(example, key: str = "code", postfix: str = ""):
    tokenizer = CppProcessor(root_folder=".")
    outputs = [tokenizer.get_tokens_and_types(code) for code in example[key]]
    return {
        f"tokens{postfix}": [output[0] for output in outputs],
        f"tags{postfix}": [output[1] for output in outputs],
    }


def java_tokenizer(example, key: str = "code", postfix: str = ""):
    tokenizer = JavaProcessor(root_folder=".")
    outputs = [tokenizer.get_tokens_and_types(code) for code in example[key]]
    return {
        f"tokens{postfix}": [output[0] for output in outputs],
        f"tags{postfix}": [output[1] for output in outputs],
    }
