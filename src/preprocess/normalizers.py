import re


def normalize_fn(example, key: str = "code"):
    outputs = [remove_comments(code) for code in example[key]]
    outputs = [remove_non_ascii(code) for code in outputs]
    outputs = [remove_space_before_newline(code) for code in outputs]
    outputs = [remove_empty_lines(code) for code in outputs]
    return {key: outputs}


def normalize_list_fn(example, key: str = "code"):
    normalized_code = []
    for code_lines in example[key]:
        outputs = [remove_comments(code) for code in code_lines]
        outputs = [remove_non_ascii(code) for code in outputs]
        outputs = [remove_space_before_newline(code) for code in outputs]
        outputs = [remove_empty_lines(code) for code in outputs]
        normalized_code.append(outputs)
    return {key: normalized_code}


# regex to remove empty lines
def remove_empty_lines(text):
    return re.sub(r"^$\n", "", text, flags=re.MULTILINE)


# regex to remove comments from a file
def remove_comments(text):
    def replacer(match):
        s = match.group(0)
        if s.startswith("/"):
            return " "  # note: a space and not an empty string
        else:
            return s

    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE,
    )
    return re.sub(pattern, replacer, text)


# regex to remove space before newLine character
def remove_space_before_newline(text):
    return re.sub(r"\s+$", "", text, flags=re.M)


# regex to remove non-ASCII characters
def remove_non_ascii(text):
    return re.sub(r"[^\x00-\x7f]", r"", text)
