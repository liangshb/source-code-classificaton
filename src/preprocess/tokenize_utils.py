def tokenize_fn(encode_fn, example, key: str = "func"):
    print(example)
    outputs = [encode_fn.get_tokens_and_types(code) for code in example[key]]
    return {
        "tokens": [output[0] for output in outputs],
        "tags": [output[1] for output in outputs],
    }
