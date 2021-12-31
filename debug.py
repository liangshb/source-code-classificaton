import json
import sys
import tempfile

from allennlp.commands import main


def debug():
    config_file = "configs/codexglue/cnn.jsonnet"
    overrides = json.dumps({"trainer": {"cuda_device": -1}})

    with tempfile.TemporaryDirectory() as serialization_dir:
        # Assemble the command into sys.argv
        sys.argv = [
            "allennlp",
            "train",
            config_file,
            "-s",
            serialization_dir,
            "-o",
            overrides,
        ]
        main()


if __name__ == "__main__":
    debug()
