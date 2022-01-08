from datasets import Dataset, DatasetDict


def split_pipeline(dataset: Dataset) -> DatasetDict:
    train, val_test = dataset.train_test_split(test_size=0.2).values()
    val, test = val_test.train_test_split(test_size=0.5).values()
    dataset_dict = DatasetDict()
    dataset_dict["train"] = train
    dataset_dict["validation"] = val
    dataset_dict["test"] = test
    dataset_dict["val_test"] = val_test
    return dataset_dict
