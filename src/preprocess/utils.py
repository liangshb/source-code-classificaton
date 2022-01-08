def merge_lines(example, key: str = "tokens-sym-no", out_key: str = "tokens-sym-no"):
    tokens_list = []
    for lines_tokens in example[key]:
        tokens = []
        for line_tokens in lines_tokens:
            tokens += line_tokens
        tokens_list.append(tokens)
    return {out_key: tokens_list}
