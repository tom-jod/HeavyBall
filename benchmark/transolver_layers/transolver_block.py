"""Transformer encoder block with physics-aware attention for the Transolver model."""

from torch import nn

from benchmark.transolver_layers.attn import PhysicsAttentionIrregular
from benchmark.transolver_layers.mlp import MLP


class TransolverBlock(nn.Module):
    """A Transformer encoder block implementation with physics-aware attention.

    This block implements a modified transformer architecture specialized for physics
    applications with the following components:

    1. Physics-aware self-attention with irregular point support
    2. Multi-layer perceptron (MLP) feed-forward network
    3. Layer normalization and residual connections
    4. Optional final projection layer for output dimension adjustment

    The block processes input features through self-attention and MLP pathways,
    maintaining residual connections throughout. When used as the final layer,
    it can project the features to a specified output dimension.
    """

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        dropout: float,
        act="gelu",
        mlp_ratio=4,
        last_layer=False,
        out_dim=1,
        slice_num=32,
    ):
        """Initialize the TransolverBlock.

        Args:
            num_heads (int): Number of attention heads for multi-head attention.
            hidden_dim (int): Dimension of the input and hidden features.
            dropout (float): Dropout probability for regularization.
            act (str, optional): Activation function to use in MLP. Defaults to "gelu".
            mlp_ratio (int, optional): Ratio of MLP hidden dimension to input dimension.
                Defaults to 4.
            last_layer (bool, optional): Whether this is the final transformer block.
                If True, adds a projection layer to out_dim. Defaults to False.
            out_dim (int, optional): Output dimension for final projection when last_layer=True.
                Defaults to 1.
            slice_num (int, optional): Number of slices for irregular physics attention.
                Controls the granularity of spatial partitioning. Defaults to 32.
        """
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.attn = PhysicsAttentionIrregular(
            hidden_dim,
            heads=num_heads,
            dim_head=hidden_dim // num_heads,
            dropout=dropout,
            slice_num=slice_num,
        )
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(
            hidden_dim, int(hidden_dim * mlp_ratio), hidden_dim, n_layers=0, res=False, act=act
        )
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, fx):
        """Forward pass of the Transolver block.

        Implements a transformer block with:
        1. Self-attention with layer norm and residual connection
        2. MLP with layer norm and residual connection
        3. Optional final projection layer (if last_layer=True)

        Args:
            fx (torch.Tensor): Input features of shape [B, N, hidden_dim]
                where:
                - B: Batch size
                - N: Number of points/vertices
                - hidden_dim: Hidden dimension size

        Returns:
            torch.Tensor: Transformed features, either:
                - Shape [B, N, hidden_dim] if not last layer
                - Shape [B, N, out_dim] if last layer
        """
        fx = self.attn(self.ln_1(fx)) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        else:
            return fx
