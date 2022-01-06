import hashlib
import re

# Sets for operators
operators3 = {"<<=", ">>="}
operators2 = {
    "->",
    "++",
    "--",
    "!~",
    "<<",
    ">>",
    "<=",
    ">=",
    "==",
    "!=",
    "&&",
    "||",
    "+=",
    "-=",
    "*=",
    "/=",
    "%=",
    "&=",
    "^=",
    "|=",
}
operators1 = {
    "(",
    ")",
    "[",
    "]",
    ".",
    "+",
    "-",
    "*",
    "&",
    "/",
    "%",
    "<",
    ">",
    "^",
    "|",
    "=",
    ",",
    "?",
    ":",
    ";",
    "{",
    "}",
    "!",
    "~",
}

# keywords up to C11 and C++17; immutable set
keywords = frozenset(
    {
        "__asm",
        "__builtin",
        "__cdecl",
        "__declspec",
        "__except",
        "__export",
        "__far16",
        "__far32",
        "__fastcall",
        "__finally",
        "__import",
        "__inline",
        "__int16",
        "__int32",
        "__int64",
        "__int8",
        "__leave",
        "__optlink",
        "__packed",
        "__pascal",
        "__stdcall",
        "__system",
        "__thread",
        "__try",
        "__unaligned",
        "_asm",
        "_Builtin",
        "_Cdecl",
        "_declspec",
        "_except",
        "_Export",
        "_Far16",
        "_Far32",
        "_Fastcall",
        "_finally",
        "_Import",
        "_inline",
        "_int16",
        "_int32",
        "_int64",
        "_int8",
        "_leave",
        "_Optlink",
        "_Packed",
        "_Pascal",
        "_stdcall",
        "_System",
        "_try",
        "alignas",
        "alignof",
        "and",
        "and_eq",
        "asm",
        "auto",
        "bitand",
        "bitor",
        "bool",
        "break",
        "case",
        "catch",
        "char",
        "char16_t",
        "char32_t",
        "class",
        "compl",
        "const",
        "const_cast",
        "constexpr",
        "continue",
        "decltype",
        "default",
        "delete",
        "do",
        "double",
        "dynamic_cast",
        "else",
        "enum",
        "explicit",
        "export",
        "extern",
        "false",
        "final",
        "float",
        "for",
        "friend",
        "goto",
        "if",
        "inline",
        "int",
        "long",
        "mutable",
        "namespace",
        "new",
        "noexcept",
        "not",
        "not_eq",
        "nullptr",
        "operator",
        "or",
        "or_eq",
        "override",
        "private",
        "protected",
        "public",
        "register",
        "reinterpret_cast",
        "return",
        "short",
        "signed",
        "sizeof",
        "static",
        "static_assert",
        "static_cast",
        "struct",
        "switch",
        "template",
        "this",
        "thread_local",
        "throw",
        "true",
        "try",
        "typedef",
        "typeid",
        "typename",
        "union",
        "unsigned",
        "using",
        "virtual",
        "void",
        "volatile",
        "wchar_t",
        "while",
        "xor",
        "xor_eq",
        "NULL",
    }
)

with open("sensiAPI.txt", "r") as f:
    a = f.read().split(",")
keywords = keywords.union(a)
# holds known non-user-defined functions; immutable set
main_set = frozenset({"main"})
# arguments in main function; immutable set
main_args = frozenset({"argc", "argv"})

unchanged = keywords.union(main_args)
unchanged = unchanged.union(main_set)


def reval_symbolize(example, tokens_key: str = "tokens", postfix: str = ""):
    symbolize_tokens = []
    for tokens in example[tokens_key]:
        f_count = 1
        var_count = 1
        symbol_table = {}
        final_tokens = []
        for idx in range(len(tokens)):
            t = tokens[idx]
            if t in keywords:
                final_tokens.append(t)
            elif t in operators1:
                final_tokens.append(t)
            elif t in operators2:
                final_tokens.append(t)
            elif t in operators3:
                final_tokens.append(t)
            elif tokens[idx + 1] == "(":
                if t in keywords:
                    final_tokens.append(t + "(")
                else:
                    if t not in symbol_table.keys():
                        symbol_table[t] = "FUN" + str(f_count)
                        f_count += 1
                    final_tokens.append(symbol_table[t] + "(")
                idx += 1

            elif t.endswith("("):
                t = t[:-1]
                if t in keywords:
                    final_tokens.append(t + "(")
                else:
                    if t not in symbol_table.keys():
                        symbol_table[t] = "FUN" + str(f_count)
                        f_count += 1
                    final_tokens.append(symbol_table[t] + "(")
            elif t.endswith("()"):
                t = t[:-2]
                if t in keywords:
                    final_tokens.append(t + "()")
                else:
                    if t not in symbol_table.keys():
                        symbol_table[t] = "FUN" + str(f_count)
                        f_count += 1
                    final_tokens.append(symbol_table[t] + "()")
            elif re.match(r'".*"', t, re.DOTALL) is not None:
                final_tokens.append(
                    '"@@STRING_{}"'.format(hashlib.sha1(t.encode("utf-8")).hexdigest()[:8])
                )
            elif re.match(r"'.*'", t, re.DOTALL) is not None:
                final_tokens.append(
                    '"@@CHAR_{}"'.format(hashlib.sha1(t.encode("utf-8")).hexdigest()[:8])
                )
            elif re.match(r"^[0-9]+(\.[0-9]+)?$", t) is not None:
                final_tokens.append(t)
            elif re.match(r"^[0-9]*(\.[0-9]+)$", t) is not None:
                final_tokens.append(t)
            else:
                if t not in symbol_table.keys():
                    symbol_table[t] = "VAR" + str(var_count)
                    var_count += 1
                final_tokens.append(symbol_table[t])
        symbolize_tokens.append(final_tokens)
    return {f"tokens{postfix}": symbolize_tokens}


def symbolize_code_var_func(example, code_key: str = "code"):
    # regular expression to find function name candidates
    rx_fun = re.compile(r"\b([_A-Za-z]\w*)\b(?=\s*\()")
    # regular expression to find variable name candidates
    # rx_var = re.compile(r'\b([_A-Za-z]\w*)\b(?!\s*\()')
    rx_var = re.compile(r"\b([_A-Za-z]\w*)\b(?:(?=\s*\w+\()|(?!\s*\w+))(?!\s*\()")

    rx_str_char = re.compile(r"(STRING_|CHAR_)")

    symbolize_code = []

    for code in example[code_key]:

        # dictionary; map function name to symbol name + number
        fun_symbols = {}
        # dictionary; map variable name to symbol name + number
        var_symbols = {}

        fun_count = 1
        var_count = 1

        user_fun = rx_fun.findall(code)
        user_var = rx_var.findall(code)

        # Could easily make a "clean gadget" type class to prevent duplicate functionality
        # of creating/comparing symbol names for functions and variables in much the same way.
        # The comparison frozenset, symbol dictionaries, and counters would be class scope.
        # So would only need to pass a string list and a string literal for symbol names to
        # another function.
        for fun_name in user_fun:
            if (
                len({fun_name}.difference(main_set)) != 0
                and len({fun_name}.difference(keywords)) != 0
                and not rx_str_char.match(fun_name)
            ):
                # DEBUG
                # print('comparing ' + str(fun_name + ' to ' + str(main_set)))
                # print(fun_name + ' diff len from main is ' + str(len({fun_name}.difference(main_set))))
                # print('comparing ' + str(fun_name + ' to ' + str(keywords)))
                # print(fun_name + ' diff len from keywords is ' + str(len({fun_name}.difference(keywords))))
                ###
                # check to see if function name already in dictionary
                if fun_name not in fun_symbols.keys():
                    fun_symbols[fun_name] = "FUN" + str(fun_count)
                    fun_count += 1
                # ensure that only function name gets replaced (no variable name with same
                # identifier); uses positive lookforward
                code = re.sub(
                    r"\b(" + fun_name + r")\b(?=\s*\()",
                    fun_symbols[fun_name],
                    code,
                )

        for var_name in user_var:
            # next line is the nuanced difference between fun_name and var_name
            if (
                len({var_name}.difference(keywords)) != 0
                and len({var_name}.difference(main_args)) != 0
                and not rx_str_char.match(var_name)
            ):
                # DEBUG
                # print('comparing ' + str(var_name + ' to ' + str(keywords)))
                # print(var_name + ' diff len from keywords is ' + str(len({var_name}.difference(keywords))))
                # print('comparing ' + str(var_name + ' to ' + str(main_args)))
                # print(var_name + ' diff len from main args is ' + str(len({var_name}.difference(main_args))))
                ###
                # check to see if variable name already in dictionary
                if var_name not in var_symbols.keys():
                    var_symbols[var_name] = "VAR" + str(var_count)
                    var_count += 1
                # ensure that only variable name gets replaced (no function name with same
                # identifier); uses negative lookforward
                code = re.sub(
                    r"\b(" + var_name + r")\b(?:(?=\s*\w+\()|(?!\s*\w+))(?!\s*\()",
                    var_symbols[var_name],
                    code,
                )

        symbolize_code.append(code)
    # return the list of cleaned lines
    return {code_key: symbolize_code}


def symbolize_code_str_char_hash(example, code_key: str = "code"):
    symbolized_code = []
    rx_str = re.compile(r'".*"', re.DOTALL)
    rx_char = re.compile(r"'.*'", re.DOTALL)

    def symbolize_str(matched):
        return '"STRING_{}"'.format(hashlib.sha1(matched.group().encode("utf-8")).hexdigest()[:8])

    def symbolize_char(matched):
        return '"CHAR_{}"'.format(hashlib.sha1(matched.group().encode("utf-8")).hexdigest()[:8])

    for code in example[code_key]:
        code = re.sub(rx_str, symbolize_str, code)
        code = re.sub(rx_char, symbolize_char, code)
        symbolized_code.append(code)
    return {f"{code_key}-hash": symbolized_code}


def symbolize_code_str_char_len(example, code_key: str = "code"):
    symbolized_code = []
    rx_str = re.compile(r'".*"', re.DOTALL)
    rx_char = re.compile(r"'.*'", re.DOTALL)

    def symbolize_str(matched):
        return '"STRING_{}"'.format(len(matched.group()) - 2)

    def symbolize_char(matched):
        return '"CHAR_{}"'.format(hashlib.sha1(matched.group().encode("utf-8")).hexdigest()[:8])

    for code in example[code_key]:
        code = re.sub(rx_str, symbolize_str, code)
        code = re.sub(rx_char, symbolize_char, code)
        symbolized_code.append(code)
    return {f"{code_key}-len": symbolized_code}


def symbolize_identifier(example, tokens_key: str = "tokens", tags_key: str = "tags"):
    symbolized_code = []
    for tokens, tags in zip(example[tokens_key], example[tags_key]):
        # identifiers = [
        #     token
        #     for token, tag in zip(tokens, tags)
        #     if tag == "identifier" and {token}.difference(unchanged)
        # ]
        # identifiers = set(identifiers)
        identifiers = []
        for token, tag in zip(tokens, tags):
            if tag == "identifier" and {token}.difference(unchanged) and token not in identifiers:
                identifiers.append(token)
        identifier_map = dict(
            zip(identifiers, [f"IDENTIFIER_{idx}" for idx in range(len(identifiers))])
        )
        new_tokens = [identifier_map.get(token, token) for token in tokens]
        symbolized_code.append(new_tokens)
    return {f"{tokens_key}-sym": symbolized_code}


def symbolize_str_char_hash(example, tokens_key: str = "tokens-sym", tags_key: str = "tags"):
    symbolized_code = []
    for tokens, tags in zip(example[tokens_key], example[tags_key]):
        new_tokens = [
            "STRING_{}".format(hashlib.sha1(token.encode("utf-8")).hexdigest()[:8])
            if "string_literal" in tag
            else token
            for token, tag in zip(tokens, tags)
        ]
        new_tokens = [
            "CHAR_{}".format(hashlib.sha1(token.encode("utf-8")).hexdigest()[:8])
            if "char_literal" in tag
            else token
            for token, tag in zip(new_tokens, tags)
        ]
        symbolized_code.append(new_tokens)
    return {f"{tokens_key}-hash": symbolized_code}


def symbolize_str_char_len(example, tokens_key: str = "tokens-sym", tags_key: str = "tags"):
    symbolized_code = []
    for tokens, tags in zip(example[tokens_key], example[tags_key]):
        new_tokens = [
            f"STRING_{len(token) - 2}" if "string_literal" in tag else token
            for token, tag in zip(tokens, tags)
        ]
        new_tokens = [
            "CHAR_{}".format(hashlib.sha1(token.encode("utf-8")).hexdigest()[:8])
            if "char_literal" in tag
            else token
            for token, tag in zip(new_tokens, tags)
        ]
        symbolized_code.append(new_tokens)
    return {f"{tokens_key}-len": symbolized_code}
