import os

import pandas as pd
from datasets import Dataset, load_from_disk


def create_meta_sysevr():
    dataset_path = "../../data/sysevr"
    dataset_names = ["API function call", "Arithmetic expression", "Pointer usage", "Array usage"]

    for name in dataset_names:
        print(f"processing {name}")
        path = os.path.join(dataset_path, name)
        dataset = load_from_disk(path)
        train_df = dataset["train"].to_pandas()
        val_test_df = dataset["val_test"].to_pandas()
        dataset_df = pd.concat([train_df, val_test_df])
        dataset = Dataset.from_pandas(dataset_df)
        dataset = dataset.remove_columns(["__index_level_0__"])
        save_path = os.path.join(dataset_path, "meta", name)
        dataset.save_to_disk(save_path)


if __name__ == "__main__":
    # main()
    create_meta_sysevr()
