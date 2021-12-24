# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import re

from src.preprocess.codegen.tokenization_utils import NEWLINE_TOKEN, ind_iter
from src.preprocess.codegen.tree_sitter_processor import (
    NEW_LINE,
    TreeSitterLangProcessor,
    extract_arguments_using_parentheses,
    get_first_token_before_first_parenthesis,
)

JAVA_TOKEN2CHAR = {
    "STOKEN00": "//",
    "STOKEN01": "/*",
    "STOKEN02": "*/",
    "STOKEN03": "/**",
    "STOKEN04": "**/",
    "STOKEN05": '"""',
    "STOKEN06": "\\n",
    "STOKEN07": "\\r",
    "STOKEN08": ";",
    "STOKEN09": "{",
    "STOKEN10": "}",
    "STOKEN11": r"\'",
    "STOKEN12": r"\"",
    "STOKEN13": r"\\",
}
JAVA_CHAR2TOKEN = {value: " " + key + " " for key, value in JAVA_TOKEN2CHAR.items()}


def remove_annotation(function):
    return re.sub(r"^@ (Override|Deprecated|SuppressWarnings) (\( .*? \) )", "", function)


def extract_functions(tokenized_code):
    """Extract functions from tokenized Java code"""
    if isinstance(tokenized_code, str):
        tokens = tokenized_code.split()
    else:
        assert isinstance(tokenized_code, list)
        tokens = tokenized_code
    i = ind_iter(len(tokens))
    functions_standalone = []
    functions_class = []
    try:
        token = tokens[i.i]
    except KeyboardInterrupt:
        raise
    except RuntimeError:
        return [], []
    while True:
        try:
            # detect function
            tokens_no_newline = []
            index = i.i
            while index < len(tokens) and len(tokens_no_newline) < 3:
                index += 1
                if tokens[index].startswith(NEWLINE_TOKEN):
                    continue
                tokens_no_newline.append(tokens[index])

            if token == ")" and (
                tokens_no_newline[0] == "{"
                or (tokens_no_newline[0] == "throws" and tokens_no_newline[2] == "{")
            ):
                # go previous until the start of function
                while token not in [";", "}", "{", "*/", "ENDCOM", NEW_LINE, "\n"]:
                    i.prev()
                    token = tokens[i.i]

                if token == "*/":
                    while token != "/*":
                        i.prev()
                        token = tokens[i.i]
                    function = [token]
                    while token != "*/":
                        i.next()
                        token = tokens[i.i]
                        function.append(token)
                elif token == "ENDCOM":
                    while token != "//":
                        i.prev()
                        token = tokens[i.i]
                    function = [token]
                    while token != "ENDCOM":
                        i.next()
                        token = tokens[i.i]
                        function.append(token)
                else:
                    i.next()
                    token = tokens[i.i]
                    function = [token]

                while token != "{":
                    i.next()
                    token = tokens[i.i]
                    function.append(token)
                if token == "{":
                    number_indent = 1
                    while not (token == "}" and number_indent == 0):
                        try:
                            i.next()
                            token = tokens[i.i]
                            if token == "{":
                                number_indent += 1
                            elif token == "}":
                                number_indent -= 1
                            function.append(token)
                        except StopIteration:
                            break
                    if "static" in function[0 : function.index("{")]:
                        functions_standalone.append(remove_annotation(" ".join(function)))
                    else:
                        functions_class.append(remove_annotation(" ".join(function)))
            i.next()
            token = tokens[i.i]
        except KeyboardInterrupt:
            raise
        except RuntimeError:
            break
    return functions_standalone, functions_class


class JavaProcessor(TreeSitterLangProcessor):
    def __init__(self, root_folder):
        super().__init__(
            language="java",
            ast_nodes_type_string=["comment", "string_literal", "character_literal"],
            stokens_to_chars=JAVA_TOKEN2CHAR,
            chars_to_stokens=JAVA_CHAR2TOKEN,
            root_folder=root_folder,
        )

    def get_function_name(self, function):
        return get_first_token_before_first_parenthesis(function)

    def extract_arguments(self, function):
        return extract_arguments_using_parentheses(function)
