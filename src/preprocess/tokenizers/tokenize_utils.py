def tokenize_fn(encode_fn, example, key: str = "func", postfix: str = ""):
    outputs = [encode_fn.get_tokens_and_types(code) for code in example[key]]
    return {
        f"tokens{postfix}": [output[0] for output in outputs],
        f"tags{postfix}": [output[1] for output in outputs],
    }
