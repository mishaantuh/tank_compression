import os
import json
import torch

from .train import train


if __name__ == "__main__":
    os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning'

    config = json.load(open("config/config.json", "r"))

    print("------------------")
    print(config)
    print("------------------")

    config["device"] = torch.device(config["device"] if torch.cuda.is_available() else 'cpu')

    train(config)