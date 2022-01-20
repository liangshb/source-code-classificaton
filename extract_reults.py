import argparse
import json
import os
from datetime import datetime


def print_result(metrics):
    train_duration = datetime.strptime(metrics["training_duration"], "%H:%M:%S.%f")
    hour = train_duration.hour
    minute = train_duration.minute
    second = train_duration.second
    time_pre_epoch = (hour * 3600 + minute * 60 + second) / metrics["epoch"]
    print("Train time(pre epoch): {} s".format(time_pre_epoch))
    print("FPR: {}".format(1 - metrics["best_validation_spec"]))
    print("FNR: {}".format(1 - metrics["best_validation_recall"]))
    print("Acc: {}".format(metrics["best_validation_accuracy"]))
    print("Pre: {}".format(metrics["best_validation_precision"]))
    print("F1: {}".format(metrics["best_validation_f1"]))
    print("MCC: {}".format(metrics["best_validation_mcc"]))


def main(path):
    for dirname in os.listdir(path):
        metrics_path = os.path.join(path, dirname, "metrics.json")
        print(f"=================={dirname}=======================")
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        print_result(metrics)


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("path", type=str)
    args = args_parser.parse_args()
    main(args.path)
