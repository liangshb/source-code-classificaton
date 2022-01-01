from os import EX_PROTOCOL
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

with open(r"data/sensiAPI.txt", "r") as f:
    a = f.read().split(",")
keywords = keywords.union(a)
# holds known non-user-defined functions; immutable set
main_set = frozenset({"main"})
# arguments in main function; immutable set
main_args = frozenset({"argc", "argv"})


def symbolize_var_func_fn(example, key: str = "code"):
    """
    change a list of code statements to their symbolic representations
    Args:
        gadget: a list of code statements

    Returns:

    """

    # dictionary; map function name to symbol name + number
    fun_symbols = {}
    # dictionary; map variable name to symbol name + number
    var_symbols = {}

    fun_count = 1
    var_count = 1

    # regular expression to find function name candidates
    rx_fun = re.compile(r"\b([_A-Za-z]\w*)\b(?=\s*\()")
    # regular expression to find variable name candidates
    # rx_var = re.compile(r'\b([_A-Za-z]\w*)\b(?!\s*\()')
    rx_var = re.compile(r"\b([_A-Za-z]\w*)\b(?:(?=\s*\w+\()|(?!\s*\w+))(?!\s*\()")

    # final cleaned gadget output to return to interface
    symbolize_code = []

    for code in example[key]:
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
    return {f"{key}-sym": symbolize_code}


def symbolize_str_char(example, key: str = "tokens-sym"):
    rx_str = re.compile(r'".*?"')
    rx_char = re.compile(r"'.*?'")
    for tokens in example[key]:
        new_tokens = [re.sub(rx_str, f"STR_{len(token)-2}", token) for token in tokens]
        new_tokens = [
            re.sub(rx_char, "CHAR_{}".format(token.strip("'")), token)
            for token in new_tokens
        ]
    return {key: new_tokens}
