import re


def tokenize_fn(example, code_key: str = "code", tokens_key: str = "tokens"):
    tokens_list = []
    for lines in example[code_key]:
        lines_tokens = []
        for line in lines:
            tokens = create_tokens(line)
            lines_tokens.append(tokens)
        tokens_list.append(lines_tokens)
    return {tokens_key: tokens_list}


def isphor(s, liter):
    m = re.search(liter, s)
    if m is not None:
        return True
    else:
        return False


def doubisphor(forward, back):
    double = (
        "->",
        "--",
        "-=",
        "+=",
        "++",
        ">=",
        "<=",
        "==",
        "!=",
        "*=",
        "/=",
        "%=",
        "/=",
        "&=",
        "^=",
        "||",
        "&&",
        ">>",
        "<<",
    )
    string = forward + back

    if string in double:
        return True
    else:
        return False


def trisphor(s, t):
    if (s == ">>") | (s == "<<") and (t == "="):
        return True
    else:
        return False


def create_tokens(sentence):
    phla = r"[^_a-zA-Z0-9]"
    space = r"\s"
    spa = r""
    string = []
    j = 0
    i = 0

    while i < len(sentence):
        if isphor(sentence[i], space):
            if i > j:
                string.append(sentence[j:i])
                j = i + 1
            else:
                j = i + 1

        elif isphor(sentence[i], phla):
            if (i + 1 < len(sentence)) and isphor(sentence[i + 1], phla):
                m = doubisphor(sentence[i], sentence[i + 1])

                if m:
                    string1 = sentence[i] + sentence[i + 1]

                    if (i + 2 < len(sentence)) and (isphor(sentence[i + 2], phla)):
                        if trisphor(string1, sentence[i + 2]):
                            string.append(sentence[j:i])
                            string.append(sentence[i] + sentence[i + 1] + sentence[i + 2])
                            j = i + 3
                            i = i + 2

                        else:
                            string.append(sentence[j:i])
                            string.append(sentence[i] + sentence[i + 1])
                            string.append(sentence[i + 2])
                            j = i + 3
                            i = i + 2

                    else:
                        string.append(sentence[j:i])
                        string.append(sentence[i] + sentence[i + 1])
                        j = i + 2
                        i = i + 1

                else:
                    string.append(sentence[j:i])
                    string.append(sentence[i])
                    string.append(sentence[i + 1])
                    j = i + 2
                    i = i + 1

            else:
                string.append(sentence[j:i])
                string.append(sentence[i])
                j = i + 1

        i = i + 1

    count = 0
    count1 = 0
    sub0 = "\r"

    if sub0 in string:
        string.remove("\r")

    for sub1 in string:
        if sub1 == " ":
            count1 = count1 + 1

    for j in range(count1):
        string.remove(" ")

    for sub in string:
        if sub == spa:
            count = count + 1

    for i in range(count):
        string.remove("")

    return string
