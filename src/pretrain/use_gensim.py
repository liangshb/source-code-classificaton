import os
from argparse import ArgumentParser

from datasets import load_from_disk
from gensim.models import FastText, Word2Vec


def main(dataset_path):
    pretrain_types = {"fasttext": FastText, "word2vec": Word2Vec}
    # token_types = {"tokens-sym-no"}
    token_types = {"merged-tokens-sym"}
    embedding_dims = (32, 64)

    train_corpus = load_from_disk(os.path.join(dataset_path, "train"))
    embedding_path = os.path.join(dataset_path, "embedding")
    os.makedirs(embedding_path, exist_ok=True)

    for k, v in pretrain_types.items():
        for t in token_types:
            print(f"Pretrain type: {k}, Token type: {t}")
            for dim in embedding_dims:
                print(f"Pretrain: {k}_{dim}_{t}")
                model = v(
                    sentences=train_corpus[t],
                    vector_size=dim,
                    min_count=1,
                    epochs=10,
                )
                model.wv.save_word2vec_format(
                    os.path.join(os.path.join(embedding_path, f"{k}_{dim}_{t}.txt")),
                )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("path", help="Path of dataset root")
    args = parser.parse_args()
    main(args.path)
