from argparse import ArgumentParser

from allennlp.interpret.attackers import Attacker, Hotflip, InputReduction

from src.predictors.cls_predictor import CLSPredictor
from src.preprocess.codegen.cpp_processor import CppProcessor
from src.preprocess.codegen.java_processor import JavaProcessor
from src.preprocess.normalizers import (
    remove_comments,
    remove_empty_lines,
    remove_space_before_newline,
)


def main():
    parser = ArgumentParser()
    parser.add_argument("code", type=str, help="Enter code to predictor and interpret")
    parser.add_argument(
        "-l", "language", type=str, help="the language of code: cpp or java", default="cpp"
    )
    parser.add_argument(
        "-p", "path", type=str, default="logs/sysevr/cnn_highway_base", help="model path"
    )
    parser.add_argument(
        "-a", "attack", type=str, help="hotflip, input_reduction", default="input_reduction"
    )
    args = parser.parse_args()

    if args.language == "cpp":
        processor = CppProcessor(root_folder="src/preprocess")
    elif args.language == "java":
        processor = JavaProcessor(root_folder="src/preprocess")
    else:
        print(f"Don't support this language: {args.language}")

    predictor = CLSPredictor.from_path(args.path, predictor_name="predictor")

    # process raw code to tokens
    code = remove_comments(args.code)
    code = remove_space_before_newline(code)
    code = remove_empty_lines(code)
    outputs = processor.get_tokens_and_types(code)
    tokens = [output[0] for output in outputs]

    # get predictor result
    res = predictor.predict(tokens)
    print(f"Prediction result: {res}")

    # interpret result
    attack_type = {
        "hotflip": Hotflip,
        "input_reduction": InputReduction,
    }
    attacker: Attacker = attack_type[args.attack](predictor)
    interpretation = attacker.attack_from_json({"tokens": tokens}, "tokens", "grad_input_1")
    print(interpretation)


if __name__ == "__main__":
    main()
