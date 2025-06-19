from flwr.common.typing import FitRes
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    parameters_to_ndarrays,
)

import os 
import pandas as pd
from typing import List, Tuple, Dict, Union, Optional

from utils_models import LogisticRegression
from utils_general import set_weights, test

class FedAvgWithStragglerDrop(FedAvg):
    """Custom FedAvg which discards updates from stragglers."""
    def __init__(self, *args, testloader, config, device, **kwargs):
        super().__init__(*args, **kwargs)
        self.exp_name = config.exp_name
        self.testloader = testloader
        self.config = config 
        self.device = device
        self.result = {"round": [], "test_loss": [], "test_accuracy": []}

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ):
        """Discard all the models sent by the clients that were stragglers."""
        # Record which client was a straggler in this round
        stragglers_mask = [res.metrics["is_straggler"] for _, res in results]

        # keep those results that are not from stragglers
        results = [res for i, res in enumerate(results) if not stragglers_mask[i]]

        # call the parent `aggregate_fit()` (i.e. that in standard FedAvg)
        return super().aggregate_fit(server_round, results, failures)
    

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate global model parameters using an evaluation function."""

        net = LogisticRegression(self.config.num_classes).to(self.device)
        set_weights(net, parameters_to_ndarrays(parameters))    
        loss, accuracy = test(net, self.testloader, self.device)

        if server_round != 0:  
            self.result['round'].append(server_round)
            self.result["test_loss"].append(loss)
            self.result["test_accuracy"].append(accuracy)

        print(f"test_loss: {loss} - test_acc: {accuracy}")

        if server_round == self.config.num_server_round:
            if not os.path.exists('result'): 
                os.makedirs('result', exist_ok=True)
            df = pd.DataFrame(self.result)
            df.to_csv(f"result/{self.exp_name}.csv", index=False)

        return float(loss), {"accuracy": accuracy}