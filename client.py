import numpy as np
import torch
from torch.utils.data import DataLoader

from flwr.client import NumPyClient
from flwr.common.typing import NDArrays, Scalar

from utils_general import get_weights, instantiate_model, set_weights, test, train, context_to_easydict
from utils_dataset import load_data 

class FlowerClient(NumPyClient):  # pylint: disable=too-many-instance-attributes
    """Standard Flower client for CNN training."""

    def __init__(
        self,
        net: torch.nn.Module,
        trainloader: DataLoader,
        valloader: DataLoader,
        num_epochs: int,
        learning_rate: float,
        straggler_schedule: np.ndarray,
        configs: dict,
    ):  # pylint: disable=too-many-arguments
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.straggler_schedule = straggler_schedule
        self.net.to(self.device)
        self.configs = configs

    def fit(self, parameters, config):
        """Implement distributed fit function for a given client."""
        set_weights(self.net, parameters)

        # At each round check if the client is a straggler,
        # if so, train less epochs (to simulate partial work)
        # if the client is told to be dropped (e.g. because not using
        # FedProx in the server), the fit method returns without doing
        # training.
        # This method always returns via the metrics (last argument being
        # returned) whether the client is a straggler or not. This info
        # is used by strategies other than FedProx to discard the update.
        if (
            self.straggler_schedule[int(config["current_round"]) - 1]
            and self.num_epochs > 1
        ):
            num_epochs = np.random.randint(1, self.num_epochs)
            if self.configs.drop_straggler:
                # return without doing any training.
                # The flag in the metric will be used to tell the strategy
                # to discard the model upon aggregation
                return (
                    get_weights(self.net),
                    len(self.trainloader),
                    {"is_straggler": True},
                )

        else:
            num_epochs = self.num_epochs

        train(
            self.net,
            self.trainloader,
            self.device,
            epochs=num_epochs,
            learning_rate=self.learning_rate,
            proximal_mu=float(self.configs.mu),
        )

        return get_weights(self.net), len(self.trainloader), {"is_straggler": False}

    def evaluate(
        self, parameters: NDArrays, config: dict[str, Scalar]
    ) -> tuple[float, int, dict]:
        """Implement distributed evaluation for a given client."""
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}




