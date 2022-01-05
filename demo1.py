import argparse
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=str)
    parser.add_argument("-p", "--predictor", type=str, default="predictor")
    parser.add_argument("-f", "--field-name", type=str, default="tokens")
    parser.add_argument("-t", "--title", type=str, default="Demo")
    parser.add_argument("-s", "--static-dir", type=str)
    args = parser.parse_args()
    command = [
        "allennlp",
        "serve",
        "--archive-path",
        f"{args.archive}",
        "--predictor",
        f"{args.predictor}",
        "--field-name",
        f"{args.field_name}",
        "--title",
        f"{args.title}",
    ]
    if args.static_dir:
        command.append("--static-dir")
        command.append(f"{args.static_dir}")
    subprocess.call(command)


if __name__ == "__main__":
    main()
