from argparse import ArgumentParser

from allennlp.interpret.saliency_interpreters import (
    IntegratedGradient,
    SaliencyInterpreter,
    SimpleGradient,
    SmoothGradient,
)

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
        "-g", "gradient", type=str, help="simple, integrated, smooth", default="simple"
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
    gradient_type = {
        "simple": SimpleGradient,
        "integrated": IntegratedGradient,
        "smooth": SmoothGradient,
    }
    interpreter: SaliencyInterpreter = gradient_type[args.gradient](predictor)
    interpretation = interpreter.saliency_interpret_from_json({"tokens": tokens})
    print(interpretation)


if __name__ == "__main__":
    main()
