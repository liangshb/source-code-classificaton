import hashlib
import os
import re

import pandas as pd
from datasets import Dataset

from src.preprocess.normalizers import remove_comments, remove_non_ascii
from src.preprocess.pipeline import split_pipeline
from src.preprocess.symbolizers.symbolizer import keywords, main_args, main_set
from src.preprocess.tokenizers.tree_sitter_tokenizer import cpp_tokenizer


def merge_lines(example, key: str = "nodes-line", out_key: str = "code"):
    code = [" ".join(lines) for lines in example[key]]
    return {out_key: code}


def split_tokenize(example, key: str = "code", out_key: str = "tokens"):
    tokens = [code.split() for code in example[key]]
    return {out_key: tokens}


def string_char_hash(example, key: str = "nodes-line", out_key: str = "nodes-line-hash"):
    new_nodes = []
    rx_str = re.compile(r'".*"', re.DOTALL)
    rx_char = re.compile(r"'.*'", re.DOTALL)

    def symbolize_str(matched):
        return '"@@STRING_{}"'.format(
            hashlib.sha1(matched.group().encode("utf-8")).hexdigest()[:8]
        )

    def symbolize_char(matched):
        return '"@@CHAR_{}"'.format(hashlib.sha1(matched.group().encode("utf-8")).hexdigest()[:8])

    for lines in example[key]:
        new_lines = []

        for line in lines:
            line = re.sub(rx_str, symbolize_str, line)
            line = re.sub(rx_char, symbolize_char, line)
            new_lines.append(line)
        new_nodes.append(new_lines)
    return {out_key: new_nodes}


def string_char_len(example, key: str = "nodes-line", out_key: str = "nodes-line-len"):
    new_nodes = []
    rx_str = re.compile(r'".*"', re.DOTALL)
    rx_char = re.compile(r"'.*'", re.DOTALL)

    def symbolize_str(matched):
        return '"@@STRING_{}"'.format(len(matched.group()) - 2)

    def symbolize_char(matched):
        return '"@@CHAR_{}"'.format(hashlib.sha1(matched.group().encode("utf-8")).hexdigest()[:8])

    for lines in example[key]:
        new_lines = []

        for line in lines:
            line = re.sub(rx_str, symbolize_str, line)
            line = re.sub(rx_char, symbolize_char, line)
            new_lines.append(line)
        new_nodes.append(new_lines)
    return {out_key: new_nodes}


def symbolize_func_var(example, nodes_key: str = "nodes-line-hash"):
    new_nodes = []
    for lines in example[nodes_key]:
        fun_symbols = {}
        var_symbols = {}
        fun_count = 1
        var_count = 1

        # regular expression to catch multi-line comment
        # rx_comment = re.compile('\*/\s*$')
        # regular expression to find function name candidates
        rx_fun = re.compile(r"\b([_A-Za-z]\w*)\b(?=\s*\()")
        # regular expression to find variable name candidates
        # rx_var = re.compile(r'\b([_A-Za-z]\w*)\b(?!\s*\()')
        rx_var = re.compile(r"\b([_A-Za-z]\w*)\b(?:(?=\s*\w+\()|(?!\s*\w+))(?!\s*\()")

        rx_str_char = re.compile(r"(STRING_|CHAR_)")

        # final cleaned gadget output to return to interface
        new_lines = []

        for line in lines:
            # replace any non-ASCII characters with empty string
            ascii_line = re.sub(r"[^\x00-\x7f]", r"", line)

            # return, in order, all regex matches at string list; preserves order for semantics
            user_fun = rx_fun.findall(ascii_line)
            user_var = rx_var.findall(ascii_line)

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
                    ascii_line = re.sub(
                        r"\b(" + fun_name + r")\b(?=\s*\()",
                        fun_symbols[fun_name],
                        ascii_line,
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
                    ascii_line = re.sub(
                        r"\b(" + var_name + r")\b(?:(?=\s*\w+\()|(?!\s*\w+))(?!\s*\()",
                        var_symbols[var_name],
                        ascii_line,
                    )
            new_lines.append(ascii_line)
        new_nodes.append(new_lines)
    return {nodes_key: new_nodes}


def normalize_statement_list(example, key: str = "nodes-line"):
    new_nodes_line = []
    for lines in example[key]:
        new_lines = [remove_comments(line) for line in lines]
        new_lines = [remove_non_ascii(line) for line in new_lines]
        new_nodes_line.append(new_lines)
    return {key: new_nodes_line}


def deepwukong_pipeline(dataset: Dataset):
    print("=================tokens-sym-no===================")
    print("normal lines")
    dataset = dataset.map(
        lambda example: normalize_statement_list(example, "nodes-line"), batched=True
    )
    print("merge lines")
    dataset = dataset.map(
        lambda example: merge_lines(example, "nodes-line-sym", "code-sym-no"),
        batched=True,
    )
    print("symbolize tokenize")
    dataset = dataset.map(
        lambda example: cpp_tokenizer(example, "code-sym-no", "-sym-no"),
        batched=True,
    )
    print("=================tokens-sym-hash===================")
    dataset = dataset.map(
        lambda example: string_char_hash(example, "nodes-line", "nodes-sym-hash"),
        batched=True,
    )
    dataset = dataset.map(
        lambda example: symbolize_func_var(example, "nodes-sym-hash"), batched=True
    )
    dataset = dataset.map(
        lambda example: merge_lines(example, "nodes-sym-hash", "code-sym-hash"),
        batched=True,
    )
    dataset = dataset.map(
        lambda example: cpp_tokenizer(example, "code-sym-hash", "-sym-hash"),
        batched=True,
    )
    print("=================tokens-sym-len===================")
    dataset = dataset.map(
        lambda example: string_char_len(example, "nodes-line", "nodes-sym-len"),
        batched=True,
    )
    dataset = dataset.map(
        lambda example: symbolize_func_var(example, "nodes-sym-len"), batched=True
    )
    dataset = dataset.map(
        lambda example: merge_lines(example, "nodes-sym-len", "code-sym-len"),
        batched=True,
    )
    dataset = dataset.map(
        lambda example: cpp_tokenizer(example, "code-sym-len", "-sym-len"),
        batched=True,
    )

    return dataset


def read_file(file_path: str, vul_label: int, vul_str: str):
    df = pd.read_json(file_path)
    df = df.rename(columns={"target": "label"})
    df["vul_label"] = df["label"]
    df.loc[df["vul_label"] == 1, "vul_label"] = vul_label
    df["vul_str"] = vul_str
    nv = df.loc[df["label"] == 0, "label"].count().item()
    v = df.loc[df["label"] == 1, "label"].count().item()
    print("Positive: {}, Negative: {}".format(v, nv))
    return df


def preprocess_deepwukong(data_dir: str, dataset_name: str = "deepwukong"):
    dataset_path = os.path.join(data_dir, dataset_name)
    raw_data_path = os.path.join(dataset_path, "raw_data")
    vul_files = [file for file in os.listdir(raw_data_path) if file.endswith(".json")]

    for vul_label, file in enumerate(vul_files):
        print(f"Processing {file}")
        file_path = os.path.join(raw_data_path, file)
        vul_str = file.split(".")[0]
        vul_df = read_file(file_path, vul_label, vul_str)
        save_path = os.path.join(dataset_path, vul_str)
        if not os.path.exists(os.path.join(save_path, "dataset_dict.json")):
            os.makedirs(save_path, exist_ok=True)
            dataset = Dataset.from_pandas(vul_df)
            dataset = deepwukong_pipeline(dataset)
            dataset_dict = split_pipeline(dataset)
            dataset_dict.save_to_disk(save_path)
            print(f"{vul_str} dataset dict saved")


def main():
    data_dir = "../../data"
    dataset_name = ["deepwukong"]
    for name in dataset_name:
        preprocess_deepwukong(data_dir, name)


if __name__ == "__main__":
    main()
