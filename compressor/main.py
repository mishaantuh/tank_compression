import os
import json
import torch
import wandb

from train import fit, init_experiment


if __name__ == "__main__":
    os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning'

    config = json.load(open("config/local_config.json", "r"))
    init_experiment(config)

    print("------------------")
    print(config)
    print("------------------")

    config["device"] = torch.device(config["device"] if torch.cuda.is_available() else 'cpu')

    if config["wandb_enable"]:
        wandb.init(project=config["wandb_project"], entity=config["wandb_entity"], name=config["name_run"])

    fit(config)
