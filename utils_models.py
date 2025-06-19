import torch 
from torch import nn 

class LogisticRegression(nn.Module):
    """A network for logistic regression using a single fully connected layer.

    As described in the Li et al., 2020 paper :

    [Federated Optimization in Heterogeneous Networks] (

    https://arxiv.org/pdf/1812.06127.pdf)
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(28 * 28, num_classes)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input Tensor that will pass through the network

        Returns
        -------
        torch.Tensor
            The resulting Tensor after it has passed through the network
        """
        output_tensor = self.linear(torch.flatten(input_tensor, 1))
        return output_tensor