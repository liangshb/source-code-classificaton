import logging
import os

from codegen.cpp_processor import CppProcessor
from datasets import Value, load_dataset, load_from_disk
from normalizers import remove_comments, remove_empty_lines, remove_space_before_newline
from tokenize_utils import tokenize_fn

log = logging.getLogger(__name__)


def get_codexglue(data_dir: str):
    dataset = load_dataset("code_x_glue_cc_defect_detection", cache_dir=data_dir)
    dataset = dataset.remove_columns(["id", "project", "commit_id"])
    dataset = dataset.rename_column("target", "label")
    label_feature = Value("int8")
    dataset = dataset.cast_column("label", label_feature)
    return dataset


def tokenize_codexglue(data_dir: str):
    save_path = os.path.join(data_dir, "code_x_glue_cc_defect_detection", "tokenized")
    if not os.path.exists(os.path.join(save_path, "train")):
        os.makedirs(save_path, exist_ok=True)
        dataset = get_codexglue(data_dir)
        log.info("Processing dataset")

        # normalize
        dataset = dataset.map(lambda example: {"func": remove_comments(example["func"])})
        dataset = dataset.map(
            lambda example: {"func": remove_space_before_newline(example["func"])}
        )
        dataset = dataset.map(lambda example: {"func": remove_empty_lines(example["func"])})

        # tokenize
        encode_fn = CppProcessor(
            root_folder=os.path.join(data_dir, "..", "src", "preprocess"),
        )
        dataset = dataset.map(lambda example: tokenize_fn(encode_fn, example), batched=True)
        os.makedirs(save_path, exist_ok=True)
        log.info("Processing done")

        dataset.save_to_disk(save_path)
        log.info("Saving done")
    else:
        log.info("Loading from previous saved")
        _ = load_from_disk(save_path)
        pass


def main():
    root_path = "../../"
    print(os.path.abspath(os.path.join(root_path, "data")))
    tokenize_codexglue(os.path.join(root_path, "data"))


if __name__ == "__main__":
    main()
