import os
from itertools import chain
from torchtext.vocab import build_vocab_from_iterator
from datasets import load_from_disk


def add_indices(vocab, example, key: str = "merged-tokens-sym"):
    indices = [vocab(sample) for sample in example[key]]
    return {
        "indices": indices
    }


def main():
    dataset_path = "../../data/sysevr/meta"
    train_set = ["API function call", "Arithmetic expression", "Pointer usage"]
    val_set = ["Array usage"]
    test_set = ["Array usage"]
    # build vocab
    tokens_list = []
    for dataset_name in train_set:
        print("phase1: {}".format(dataset_name))
        dataset = load_from_disk(os.path.join(dataset_path, dataset_name))
        tokens_list.append(dataset["merged-tokens-sym"])
    vocab = build_vocab_from_iterator(chain(*tokens_list), min_freq=1, specials=["<pad>", "<unk>"])
    vocab.set_default_index(vocab(["<unk>"])[0])
    # add indices
    for dataset_name in chain(train_set, val_set):
        print("phase2: {}".format(dataset_name))
        dataset = load_from_disk(os.path.join(dataset_path, dataset_name))
        dataset = dataset.map(lambda example: add_indices(vocab, example), batched=True)
        dataset.save_to_disk(os.path.join(dataset_path, "meta_idx", dataset_name))


if __name__ == "__main__":
    main()
