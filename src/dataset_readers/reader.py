from typing import Dict, Iterable, List, Union

from allennlp.data import DatasetReader, Instance, Token
from allennlp.data.fields import Field, LabelField, TextField
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from datasets import load_from_disk


@DatasetReader.register("reader")
class Reader(DatasetReader):
    def __init__(
        self,
        token_indexers: Dict[str, TokenIndexer] = None,
        max_sequence_length: int = None,
        skip_label_indexing: bool = False,
        tokens_key: str = "tokens",
        label_key: str = "label",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.token_indexers = token_indexers or {
            "tokens": SingleIdTokenIndexer(namespace="tokens")
        }
        self.max_sequence_length = max_sequence_length
        self.skip_label_indexing = skip_label_indexing
        self.token_key = tokens_key
        self.label_key = label_key

    def text_to_instance(self, tokens: List[str], label: Union[str, int] = None) -> Instance:
        fields: Dict[str, Field] = {}
        if self.max_sequence_length is not None:
            tokens = self.truncate(tokens)
        tokens = [Token(token) for token in tokens]
        fields["tokens"] = TextField(tokens, self.token_indexers)
        if label is not None:
            fields["label"] = LabelField(label, skip_indexing=self.skip_label_indexing)
        return Instance(fields)

    def _read(self, file_path: str) -> Iterable[Instance]:
        dataset = load_from_disk(file_path)
        for sample in dataset:
            tokens = sample[self.token_key]
            label = sample.get(self.label_key)
            if label is not None:
                if self.skip_label_indexing:
                    try:
                        label = int(label)
                    except ValueError:
                        raise ValueError("Labels must be integers if skip_label_indexing is True")
                else:
                    label = str(label)
            yield self.text_to_instance(tokens, label)

    def truncate(self, items):
        """
        truncate a set of tokens using the provided sequence length
        """
        if len(items) > self._max_sequence_length:
            items = items[: self._max_sequence_length]
        return items
