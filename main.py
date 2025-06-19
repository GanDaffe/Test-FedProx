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
model_cfg = cfg.model 

# -------------------------- CLIENT FN ------------------------------------ 
def client_fn(context: Context):

    net = LogisticRegression(model_cfg.num_classes) 

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
def gen_evaluate_fn(
    testloader: DataLoader,
    device: torch.device,
    run_config: dict,
) -> Callable[
    [int, NDArrays, dict[str, Scalar]], tuple[float, dict[str, Scalar]] | None
]:
    """Generate the function for centralized evaluation.

    Parameters
    ----------
    testloader : DataLoader
        The dataloader to test the model with.
    device : torch.device
        The device to test the model on.

    Returns
    -------
    Callable[ [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]] ]
        The centralized evaluation function.
    """

    def evaluate(
        server_round: int, parameters_ndarrays: NDArrays, config: dict[str, Scalar]
    ) -> tuple[float, dict[str, Scalar]] | None:
        # pylint: disable=unused-argument
        """Use the entire MNIST test set for evaluation."""
        net = LogisticRegression(run_config.num_classes)
        set_weights(net, parameters_ndarrays)
        net.to(device)

        # We could compile the model here but we are not going to do it because
        # running test() is so lightweight that the overhead of compiling the model
        # negate any potential speedup. Please note this is specific to the model and
        # dataset used in this baseline. In general, compiling the model is worth it

        loss, accuracy = test(net, testloader, device=device)
        # return statistics
        return loss, {"accuracy": accuracy}

    return evaluate


# Define metric aggregation function
def weighted_average(metrics: list[tuple[int, Metrics]]) -> Metrics:
    """Do weighted average of accuracy metric."""
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * float(m["accuracy"]) for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

def get_on_fit_config():
        def fit_config_fn(server_round: int):
            # resolve and convert to python dict
            return {"current_round": server_round}

        return fit_config_fn

testloader = prepare_test_loader(
        dataset_cfg  # pylint: disable=E1101
    )

ndarrays = get_weights(LogisticRegression(model_cfg.num_classes))
parameters = ndarrays_to_parameters(ndarrays)
evaluate_fn = gen_evaluate_fn(
     testloader, 
     device=DEVICE, 
     run_config=model_cfg
)

strategy = FedAvgWithStragglerDrop(
    fraction_fit=float(general_cfg.fraction_fit),
    fraction_evaluate=general_cfg.fraction_evaluate,
    min_available_clients=general_cfg.min_available_clients,
    initial_parameters=parameters,
    on_fit_config_fn=get_on_fit_config(),
    evaluate_fn=evaluate_fn,
)

client_resources = {"num_cpus": 2, "num_gpus": 0.2} if DEVICE == "cuda" else {"num_cpus": 1, "num_gpus": 0.0}
fl.simulation.start_simulation(
            client_fn           = client_fn,
            num_clients         = general_cfg.num_clients,
            config              = fl.server.ServerConfig(num_rounds=general_cfg.num_server_round),
            strategy            = strategy,
            client_resources     = client_resources
        )