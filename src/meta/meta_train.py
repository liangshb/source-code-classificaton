#!/usr/bin/env python3

import argparse
import os
import random
from itertools import chain
from typing import List, Tuple

import learn2learn as l2l
from allennlp.data import Instance
from allennlp.data.data_loaders import DataLoader, MultiProcessDataLoader
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from datasets import load_from_disk
from torch.utils.data import Dataset

from src.dataset_readers.reader import Reader
from src.models.cls_model import Classifier
from src.modules.cnn_highway_mask import CnnHighwayMaskEncoder

model_conf = {
    "embedding_dim": 64,
    "num_filters": 128,
    "ngram_filter_sizes": (5, 6, 7, 8),
    "num_highway": 2,
    "projection_dim": 64,
    "activation": "relu",
    "projection_location": "after_highway",
    "do_layer_norm": True,
}
batch_size = 128
dropout = 0.1
min_count = {"tokens": 1}


def build_dataset_reader() -> DatasetReader:
    return Reader(tokens_key="merged-tokens-sym", label_key="label")


def build_vocab(data_loaders: List[DataLoader]) -> Vocabulary:
    print("Building the vocabulary")
    return Vocabulary.from_instances(
        chain(*[loader.iter_instances() for loader in data_loaders]), min_count=min_count
    )


def build_model(vocab: Vocabulary) -> Model:
    print("Building the model")
    vocab_size = vocab.get_vocab_size()
    embedder = BasicTextFieldEmbedder(
        {"tokens": Embedding(model_conf["embedding_dim"], vocab_size, padding_index=0)}
    )
    encoder = CnnHighwayMaskEncoder(**model_conf)
    return Classifier(vocab, embedder, encoder, dropout=dropout)


def build_data_loaders(
    reader: DatasetReader, train_data_path: str
) -> Tuple[DataLoader, DataLoader]:
    train_loader = MultiProcessDataLoader(
        reader, train_data_path, batch_size=batch_size, shuffle=False
    )
    return train_loader


class DatasetWrapper(Dataset):
    def __init__(self, dataset_path):
        self.data = load_from_disk(dataset_path)

    def __getitem__(self, item):
        sample = self.data[item]
        return sample["merged-tokens-sym"], sample["label"]

    def __len__(self):
        return len(self.data)


def setup():
    dataset_path = "../../data/sysevr/meta"
    train = ["API_function_call", "Arithmetic_expression", "Pointer_usage"]
    test = ["Array_usage"]
    reader = build_dataset_reader()
    data_loaders = []
    for name in train:
        train_path = os.path.join(dataset_path, name)
        data_loader = build_data_loaders(reader, train_path)
        data_loaders.append(data_loader)
    vocab = build_vocab(data_loaders)
    model = build_model(vocab)
    return model, vocab


def main():
    dataset_path = "../../data/sysevr/meta"
    train = ["API_function_call", "Arithmetic_expression", "Pointer_usage"]
    test = ["Array_usage"]
    train_set_list = []
    for name in train:
        train_path = os.path.join(dataset_path, name)
        train_set_list.append(l2l.data.MetaDataset(DatasetWrapper(train_path)))


if __name__ == "__main__":
    setup()
