import logging
import os
from argparse import ArgumentParser

from datasets import load_from_disk
from gensim.models import FastText, Word2Vec


def main(dataset_root):
    pretrain_types = {"fasttext": FastText, "word2vec": Word2Vec}
    embedding_dims = (50, 100, 150)

    train_corpus = load_from_disk(os.path.join(dataset_root, "tokenized", "train"))
    embedding_path = os.path.join(dataset_root, "embedding")
    os.makedirs(embedding_path, exist_ok=True)

    for k, v in pretrain_types.items():
        print(f"Pretrain type: {k}")
        for dim in embedding_dims:
            print(f"Pretrain: {k}_{dim}")
            model = v(sentences=train_corpus["tokens"], vector_size=dim, min_count=1, epochs=10)
            model.wv.save_word2vec_format(
                os.path.join(os.path.join(embedding_path, f"{k}_{dim}.txt")),
                prefix="'",
            )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("path", help="Path of dataset root")
    args = parser.parse_args()
    main(args.path)
