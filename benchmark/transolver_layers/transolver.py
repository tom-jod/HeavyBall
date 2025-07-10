"""Main Transolver model implementation for field regression tasks."""

import torch
from timm.models.layers import trunc_normal_
from torch import nn

from benchmark.transolver_layers.mlp import MLP
from benchmark.transolver_layers.transolver_block import TransolverBlock


class TransolverReg(nn.Module):
    """Transolver model for field regression tasks."""

    def __init__(
        self,
        feature_dim: int = 1,
        n_layers: int = 5,
        n_hidden: int = 256,
        dropout: float = 0.0,
        n_head: int = 8,
        act: str = "gelu",
        mlp_ratio: float = 1,
        function_dim: int = 1,
        out_dim: int = 1,
        slice_num: int = 32,
        ref: int = 8,
    ):
        """Initialize the Transolver model for field regression.

        This constructs a Transolver model with the specified architecture parameters.
        The model consists of:
        1. A preprocessing MLP to handle input features
        2. A stack of Transolver blocks for feature transformation
        3. A learnable placeholder feature for augmenting input when needed

        Args:
            feature_dim (int, optional): Dimension of input features per point.
                Defaults to 1.
            n_layers (int, optional): Number of Transolver blocks in the model.
                More layers allow learning more complex functions. Defaults to 5.
            n_hidden (int, optional): Hidden dimension size used throughout the model.
                Controls model capacity and representation power. Defaults to 256.
            dropout (float, optional): Dropout probability for regularization.
                Applied in attention and MLP layers. Defaults to 0.0.
            n_head (int, optional): Number of attention heads in each Transolver block.
                Each head processes different aspects of the input. Defaults to 8.
            act (str, optional): Activation function used in MLPs.
                Defaults to "gelu".
            mlp_ratio (float, optional): Ratio of MLP hidden dimension to input dimension
                in Transolver blocks. Controls feed-forward network capacity.
                Defaults to 1.
            function_dim (int, optional): Dimension of function features that may be
                combined with input features. Defaults to 1.
            out_dim (int, optional): Output dimension of the final prediction.
                Defaults to 1.
            slice_num (int, optional): Number of slices for irregular physics attention.
                Controls spatial partitioning granularity. Defaults to 32.
            ref (int, optional): Reference parameter for model configuration.
                Defaults to 8.
        """
        super().__init__()
        self.ref = ref
        self.preprocess = MLP(
            function_dim + feature_dim, n_hidden * 2, n_hidden, n_layers=0, res=False, act=act
        )

        self.n_hidden = n_hidden
        self.feature_dim = feature_dim

        self.blocks = nn.ModuleList(
            [
                TransolverBlock(
                    num_heads=n_head,
                    hidden_dim=n_hidden,
                    dropout=dropout,
                    act=act,
                    mlp_ratio=mlp_ratio,
                    out_dim=out_dim,
                    slice_num=slice_num,
                    last_layer=(_ == n_layers - 1),
                )
                for _ in range(n_layers)
            ]
        )
        self.initialize_weights()
        self.placeholder = nn.Parameter((1 / (n_hidden)) * torch.rand(n_hidden, dtype=torch.float))

    def initialize_weights(self):
        """Initialize Transolver weights."""
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """Forward pass of the Transolver model for field regression.

        The forward pass consists of:
        1. Input preprocessing: Expands and reshapes input, handles 4D inputs
        2. Feature processing: Combines with placeholder features if no auxiliary features
        3. Sequential transformation: Passes through Transolver blocks

        Args:
            x (torch.Tensor): Input features tensor of shape [B, input_channels, N]
                where:
                - B: Batch size
                - input_channels: Number of input feature channels
                - N: Number of points/vertices

        Returns:
            torch.Tensor: Field prediction output of shape [B, output_dim, N]
                         or [B, N] if output_dim is 1
        """
        fx = None
        x = x[None, :, :]

        if len(x.shape) == 4:
            x = x.squeeze(1)  # Remove dimension if it's [1, 1, N, C]

        if fx is not None:
            fx = torch.cat((x, fx), -1)
            fx = self.preprocess(fx)
        else:
            fx = self.preprocess(x)
            fx = fx + self.placeholder[None, None, :]

        for block in self.blocks:
            fx = block(fx)

        return fx[0]
