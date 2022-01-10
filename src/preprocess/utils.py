def merge_lines(example, key: str = "tokens-sym-no", out_key: str = "tokens-sym-no"):
    tokens_list = []
    for lines_tokens in example[key]:
        tokens = []
        for line_tokens in lines_tokens:
            tokens += line_tokens
        tokens_list.append(tokens)
    return {out_key: tokens_list}


def merge_lines_tokens(example, key: str = "tokens-sym", out_key: str = "merged-tokens-sym"):
    merged_lines_list = []
    for lines in example[key]:
        merged_lines = []
        for line in lines:
            merged_lines += line
        merged_lines_list.append(merged_lines)
    return {out_key: merged_lines_list}
