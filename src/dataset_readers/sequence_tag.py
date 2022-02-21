import logging
from typing import Any, Dict, List

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, MetadataField, SequenceLabelField, TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token


@DatasetReader.register("sequence_tag")
class SequenceTagDatasetReader(DatasetReader):
    def __init__(
        self,
        token_indexers: Dict[str, TokenIndexer] = None,
        max_sequence_length: int = None,
        skip_label_indexing: bool = False,
        tokens_key: str = "tokens",
        tags_key: str = "tags",
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    def _read(self, file_path):
        pass

    def text_to_instance(self, tokens: List[Token], tags: List[str] = None) -> Instance:
        fields: Dict[str, Field] = {}
        sequence = TextField(tokens)
        fields["tokens"] = sequence
        if tags is not None:
            fields["tags"] = SequenceLabelField(tags, sequence)
        return Instance(fields)

    def apply_token_indexers(self, instance: Instance) -> None:
        pass
