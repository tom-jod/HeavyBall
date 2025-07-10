"""Physics-aware attention mechanisms for the Transolver model."""

import torch
from einops import rearrange
from torch import nn


class PhysicsAttentionIrregular(nn.Module):
    """Physics Attention Module."""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, slice_num=64):
        """Initialize the Physics Attention Module with irregular point support.

        This module implements a specialized attention mechanism for physics applications
        that operates on irregular point clouds or mesh structures. It uses a sliced
        attention approach where points are first grouped into learned slices before
        computing attention.

        Args:
            dim (int): Input dimension of features per point.
            heads (int, optional): Number of attention heads. Each head processes a different
                projection of the input. Defaults to 8.
            dim_head (int, optional): Dimension of each attention head. The total inner
                dimension will be heads * dim_head. Defaults to 64.
            dropout (float, optional): Dropout probability for regularization. Applied to
                attention weights and output projections. Defaults to 0.0.
            slice_num (int, optional): Number of slices to divide points into before
                computing attention. Controls the granularity of spatial partitioning.
                Defaults to 64.

        The initialization creates several components:
        - Input projections for features (x) and transformed features (fx)
        - Slice projection with orthogonal weight initialization
        - Query, Key, Value projections for attention computation
        - Output projection with dropout
        - Learnable temperature parameter for attention scaling
        """
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head**-0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)

        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_fx = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        for layer in [self.in_project_slice]:
            torch.nn.init.orthogonal_(layer.weight)  # use a principled initialization
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        """Forward function implementing a sliced attention mechanism.

        The forward pass consists of three main steps:
        1. Slice: Projects input into slice tokens using learned weights
        2. Attention: Performs attention computation among slice tokens
        3. Deslice: Projects slice tokens back to original dimension

        Args:
            x: Input tensor of shape [batch_size, num_points, channels]
               where:
               - batch_size: Number of samples in the batch
               - num_points: Number of points/vertices in each sample
               - channels: Number of input features per point

        Returns:
            Tensor of shape [batch_size, num_points, out_channels]
            containing the transformed features after slice attention
        """
        # B N C
        batch_size, n_points, n_channels = x.shape

        ### (1) Slice
        fx_mid = (
            self.in_project_fx(x)
            .reshape(batch_size, n_points, self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .contiguous()
        )  # B H N C
        x_mid = (
            self.in_project_x(x)
            .reshape(batch_size, n_points, self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .contiguous()
        )  # B H N C
        slice_weights = self.softmax(self.in_project_slice(x_mid) / self.temperature)  # B H N G
        slice_norm = slice_weights.sum(2)  # B H G
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / (
            (slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head)
        )

        ### (2) Attention among slice tokens
        q_slice_token = self.to_q(slice_token)
        k_slice_token = self.to_k(slice_token)
        v_slice_token = self.to_v(slice_token)
        dots = torch.matmul(q_slice_token, k_slice_token.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)
        attn = self.dropout(attn)
        out_slice_token = torch.matmul(attn, v_slice_token)  # B H G D

        ### (3) Deslice
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        out_x = rearrange(out_x, "b h n d -> b n (h d)")
        return self.to_out(out_x)
