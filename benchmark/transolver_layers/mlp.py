"""Multi-layer perceptron implementation with optional residual connections."""

from torch import nn

ACTIVATION = {
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU(0.1),
    "softplus": nn.Softplus,
    "ELU": nn.ELU,
    "silu": nn.SiLU,
}


class MLP(nn.Module):
    """Sequential MLP."""

    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act="gelu", res=True):
        """Initialize a sequential Multi-Layer Perceptron (MLP).

        Args:
            n_input (int): Number of input features. Dimension of the input tensor.
            n_hidden (int): Number of hidden units in each layer. Controls the model's
                capacity and representation power.
            n_output (int): Number of output features. Dimension of the output tensor.
            n_layers (int, optional): Number of hidden layers between the input and
                output layers. More layers allow learning more complex functions.
                Defaults to 1.
            act (str, optional): Activation function to use after each linear layer.
                Must be one of: "gelu", "tanh", "sigmoid", "relu", "leaky_relu",
                "softplus", "ELU", "silu". Defaults to "gelu".
            res (bool, optional): Whether to use residual connections between layers.
                If True, adds skip connections that help with gradient flow in deeper
                networks. Defaults to True.
        """
        super(MLP, self).__init__()

        if act in ACTIVATION.keys():
            act = ACTIVATION[act]
        else:
            raise NotImplementedError
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), act())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList(
            [nn.Sequential(nn.Linear(n_hidden, n_hidden), act()) for _ in range(n_layers)]
        )

    def forward(self, x):
        """Sequential MLP forward."""
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            if self.res:
                x = self.linears[i](x) + x
            else:
                x = self.linears[i](x)
        x = self.linear_post(x)
        return x
