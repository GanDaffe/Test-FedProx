from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.common.typing import NDArrays, Scalar
from collections.abc import Callable
import flwr as fl
import torch 
from torch.utils.data import DataLoader

import numpy as np 
import yaml 
from easydict import EasyDict

from client import FlowerClient
from strategy import FedAvgWithStragglerDrop
from utils_general import get_weights, set_weights, test
from utils_dataset import load_data, prepare_test_loader
from utils_models import LogisticRegression

# -------------------------- LOAD CONFIG ---------------------------------
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' 
with open("config.yaml") as f: 
    cfg = EasyDict(yaml.safe_load(f)) 

general_cfg = cfg.defaults
dataset_cfg = cfg.dataset

# -------------------------- CLIENT FN ------------------------------------ 
def client_fn(context: Context):

    net = LogisticRegression(general_cfg.num_classes) 

    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(general_cfg.num_clients)
    local_epochs = general_cfg.local_epochs
    num_rounds = general_cfg.num_server_round
    stragglers = general_cfg.straggler_fraction

    straggler_schedule = np.transpose(
        np.random.choice([0, 1], size=(num_rounds, 1), p=[1 - stragglers, stragglers])
    )[0]

    trainloader, valloader = load_data(
        dataset_config=dataset_cfg,
        partition_id=partition_id,
        num_partitions=num_partitions,
    )
    learning_rate = general_cfg.lr

    # Return Client instance
    return FlowerClient(
        net,
        trainloader,
        valloader,
        local_epochs,
        learning_rate,
        straggler_schedule,
        configs=general_cfg,
    ).to_client()

# --------------------------------- EVAL FN -----------------------------------

def get_on_fit_config():
        def fit_config_fn(server_round: int):
            # resolve and convert to python dict
            return {"current_round": server_round}

        return fit_config_fn

testloader = prepare_test_loader(
        dataset_cfg  # pylint: disable=E1101
    )

ndarrays = get_weights(LogisticRegression(general_cfg.num_classes))
parameters = ndarrays_to_parameters(ndarrays)

strategy = FedAvgWithStragglerDrop(
    fraction_fit=float(general_cfg.fraction_fit),
    fraction_evaluate=general_cfg.fraction_evaluate,
    min_available_clients=general_cfg.min_available_clients,
    initial_parameters=parameters,
    on_fit_config_fn=get_on_fit_config(),
    testloader=testloader,
    config=general_cfg, 
    device=DEVICE
)

client_resources = {"num_cpus": 2, "num_gpus": 0.2} if DEVICE == "cuda" else {"num_cpus": 1, "num_gpus": 0.0}
fl.simulation.start_simulation(
            client_fn           = client_fn,
            num_clients         = general_cfg.num_clients,
            config              = fl.server.ServerConfig(num_rounds=general_cfg.num_server_round),
            strategy            = strategy,
            client_resources     = client_resources
        )