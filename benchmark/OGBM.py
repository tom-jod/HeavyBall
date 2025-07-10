from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import torch
import torch.nn as nn
import typer
from torch.nn import functional as F
import numpy as np
import tensorflow_datasets as tfds
from functools import partial
from torch.utils.data import DataLoader
from benchmark.utils import loss_win_condition, trial
from heavyball.utils import set_torch

app = typer.Typer(pretty_exceptions_enable=False)
set_torch()
import torch._dynamo
torch._dynamo.config.suppress_errors = True
app = typer.Typer()

torch._dynamo.config.disable = True

# GraphsTuple implementation for PyTorch
class GraphsTuple:
    def __init__(self, nodes, edges, receivers, senders, globals, n_node, n_edge):
        self.nodes = nodes
        self.edges = edges
        self.receivers = receivers
        self.senders = senders
        self.globals = globals
        self.n_node = n_node
        self.n_edge = n_edge
    
    def _replace(self, **kwargs):
        new_dict = {
            'nodes': self.nodes,
            'edges': self.edges,
            'receivers': self.receivers,
            'senders': self.senders,
            'globals': self.globals,
            'n_node': self.n_node,
            'n_edge': self.n_edge
        }
        new_dict.update(kwargs)
        return GraphsTuple(**new_dict)

def scatter_sum(src: torch.Tensor,
                index: torch.Tensor,
                dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> torch.Tensor:
    """Scatter sum operation"""
    index = broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index, src)
    else:
        return out.scatter_add_(dim, index, src)

def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    """Broadcast tensor for scatter operations"""
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand(other.size())
    return src

def _make_mlp(in_dim, hidden_dims, dropout_rate, activation_fn):
    """Creates a MLP with specified dimensions."""
    layers = nn.Sequential()
    for i, dim in enumerate(hidden_dims):
        layers.add_module(f'dense_{i}',
                         nn.Linear(in_features=in_dim, out_features=dim))
        layers.add_module(f'norm_{i}', nn.LayerNorm(dim, eps=1e-6))
        layers.add_module(f'activation_fn_{i}', activation_fn())
        layers.add_module(f'dropout_{i}', nn.Dropout(dropout_rate))
        in_dim = dim
    return layers

class GraphNetwork(nn.Module):
    """Graph Network layer following algoperf implementation"""
    
    def __init__(self,
                 update_edge_fn: Optional[nn.Module] = None,
                 update_node_fn: Optional[nn.Module] = None,
                 update_global_fn: Optional[nn.Module] = None) -> None:
        super().__init__()
        self.update_edge_fn = update_edge_fn
        self.update_node_fn = update_node_fn
        self.update_global_fn = update_global_fn

    def forward(self, graph: GraphsTuple) -> GraphsTuple:
        """Apply graph network layer"""
        nodes, edges, receivers, senders, globals_, n_node, n_edge = (
            graph.nodes, graph.edges, graph.receivers, graph.senders,
            graph.globals, graph.n_node, graph.n_edge)
        
        sum_n_node = nodes.shape[0]
        
        # Get node attributes for edges
        sent_attributes = nodes[senders]
        received_attributes = nodes[receivers]
        
        # Scatter global features to edges
        global_edge_attributes = torch.repeat_interleave(globals_, n_edge, dim=0)
        
        # Update edges
        if self.update_edge_fn:
            edge_fn_inputs = torch.cat(
                [edges, sent_attributes, received_attributes, global_edge_attributes],
                dim=-1)
            edges = self.update_edge_fn(edge_fn_inputs)
        
        # Update nodes
        if self.update_node_fn:
            sent_attributes = scatter_sum(edges, senders, dim=0, dim_size=sum_n_node)
            received_attributes = scatter_sum(edges, receivers, dim=0, dim_size=sum_n_node)
            
            # Scatter global features to nodes
            global_attributes = torch.repeat_interleave(globals_, n_node, dim=0)
            
            node_fn_inputs = torch.cat(
                [nodes, sent_attributes, received_attributes, global_attributes],
                dim=-1)
            nodes = self.update_node_fn(node_fn_inputs)
        
        # Update globals
        if self.update_global_fn:
            n_graph = n_node.shape[0]
            graph_idx = torch.arange(n_graph, device=nodes.device)
            
            # Create node and edge to graph mappings
            node_gr_idx = torch.repeat_interleave(graph_idx, n_node, dim=0)
            edge_gr_idx = torch.repeat_interleave(graph_idx, n_edge, dim=0)
            
            # Aggregate nodes and edges per graph
            node_attributes = scatter_sum(nodes, node_gr_idx, dim=0, dim_size=n_graph)
            edge_attributes = scatter_sum(edges, edge_gr_idx, dim=0, dim_size=n_graph)
            
            global_fn_inputs = torch.cat([node_attributes, edge_attributes, globals_],
                                       dim=-1)
            globals_ = self.update_global_fn(global_fn_inputs)
        
        return GraphsTuple(
            nodes=nodes,
            edges=edges,
            receivers=receivers,
            senders=senders,
            globals=globals_,
            n_node=n_node,
            n_edge=n_edge)

class GNN(nn.Module):
    """Algoperf GNN implementation"""
    
    def __init__(self,
                 num_outputs: int = 128,
                 dropout_rate: Optional[float] = 0.1,
                 activation_fn_name: str = 'relu',
                 latent_dim: int = 256,
                 hidden_dims: Tuple[int] = (256,),
                 num_message_passing_steps: int = 5) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.num_message_passing_steps = num_message_passing_steps
        self.num_outputs = num_outputs
        
        if dropout_rate is None:
            dropout_rate = 0.1
        
        # Node and edge embedders (specific to OGBG workload)
        self.node_embedder = nn.Linear(in_features=9, out_features=self.latent_dim)
        self.edge_embedder = nn.Linear(in_features=3, out_features=self.latent_dim)
        
        # Activation function
        if activation_fn_name == 'relu':
            activation_fn = nn.ReLU
        elif activation_fn_name == 'gelu':
            activation_fn = partial(nn.GELU, approximate='tanh')
        elif activation_fn_name == 'silu':
            activation_fn = nn.SiLU
        else:
            raise ValueError(f'Invalid activation function name: {activation_fn_name}')
        
        # Build graph network layers
        graph_network_layers = []
        for st in range(self.num_message_passing_steps):
            # Input dimensions based on algoperf implementation
            if st == 0:
                in_dim_edge_fn = self.latent_dim * 3 + self.num_outputs
                in_dim_node_fn = self.latent_dim + self.hidden_dims[-1] * 2 + self.num_outputs
                last_in_dim = self.hidden_dims[-1] * 2 + self.num_outputs
            else:
                in_dim_edge_fn = self.hidden_dims[-1] * 4
                in_dim_node_fn = self.hidden_dims[-1] * 4
                last_in_dim = self.hidden_dims[-1] * 3
            
            graph_network_layers.append(
                GraphNetwork(
                    update_edge_fn=_make_mlp(in_dim_edge_fn,
                                           self.hidden_dims,
                                           dropout_rate,
                                           activation_fn),
                    update_node_fn=_make_mlp(in_dim_node_fn,
                                           self.hidden_dims,
                                           dropout_rate,
                                           activation_fn),
                    update_global_fn=_make_mlp(last_in_dim,
                                             self.hidden_dims,
                                             dropout_rate,
                                             activation_fn)))
        
        self.graph_network = nn.Sequential(*graph_network_layers)
        self.decoder = nn.Linear(
            in_features=self.hidden_dims[-1], out_features=self.num_outputs)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using PyTorch defaults"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=np.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / np.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)
    
    def forward(self, graph: GraphsTuple) -> torch.Tensor:
        # Initialize globals with zeros
        graph = graph._replace(
            globals=torch.zeros([graph.n_node.shape[0], self.num_outputs],
                               device=graph.n_node.device))
        
        # Embed nodes and edges
        graph = graph._replace(nodes=self.node_embedder(graph.nodes))
        graph = graph._replace(edges=self.edge_embedder(graph.edges))
        
        # Apply graph network layers
        graph = self.graph_network(graph)
        
        # Final prediction
        graph = graph._replace(globals=self.decoder(graph.globals))
        return graph.globals

class OGBGDataset:
    """OGBG-MolPCBA dataset with automatic download via TFDS"""
    
    def __init__(self, data_dir="/mnt/storage01/home/tomjodrell/ogbg", split='train'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.split = split
        
        # Automatically download dataset via TFDS
        print(f"Loading OGBG-MolPCBA dataset (split: {split})...")
        print("This will automatically download the dataset if not present locally.")
        
        tfds_split_map = {
            'train': 'train',
            'validation': 'validation',
            'test': 'test'
        }
        
        # TFDS automatically downloads if not present
        self.dataset = tfds.load(
            'ogbg_molpcba:0.1.3',
            split=tfds_split_map[split],
            data_dir=str(self.data_dir),
            shuffle_files=(split == 'train')
        )
        
        # Convert to list for easier access (limit for demo)
        self.examples = list(self.dataset)#.take(1000)
        print(f"Loaded {len(self.examples)} examples")
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.split = split
        
        # Automatically download dataset via TFDS
        print(f"Loading OGBG-MolPCBA dataset (split: {split})...")
        print("This will automatically download the dataset if not present locally.")
        
        tfds_split_map = {
            'train': 'train',
            'validation': 'validation',
            'test': 'test'
        }
        
        # TFDS automatically downloads if not present
        self.dataset = tfds.load(
            'ogbg_molpcba:0.1.3',
            split=tfds_split_map[split],
            data_dir=str(self.data_dir),
            shuffle_files=(split == 'train')
        )
        
        # Convert to list for easier access (limit for demo)
        self.examples = list(self.dataset)
        print(f"Loaded {len(self.examples)} examples")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Convert TensorFlow tensors to numpy then torch
        node_feat = torch.from_numpy(example['node_feat'].numpy()).float()
        edge_feat = torch.from_numpy(example['edge_feat'].numpy()).float()
        edge_index = torch.from_numpy(example['edge_index'].numpy()).long()
        labels = torch.from_numpy(example['labels'].numpy()).float()
        
        # Handle NaN labels (set to 0, create mask)
        nan_mask = torch.isnan(labels)
        labels = torch.nan_to_num(labels, nan=0.0)
        weights = ~nan_mask  # Valid labels have weight 1, NaN labels have weight 0
        
        return {
            'nodes': node_feat,
            'edges': edge_feat,
            'edge_index': edge_index,
            'labels': labels,
            'weights': weights
        }

def collate_graphs(batch):
    """Collate function to create GraphsTuple batches"""
    # Separate components
    all_nodes = []
    all_edges = []
    all_senders = []
    all_receivers = []
    all_labels = []
    all_weights = []
    n_nodes = []
    n_edges = []
    
    node_offset = 0
    for item in batch:
        nodes = item['nodes']
        edges = item['edges']
        edge_index = item['edge_index']
        
        all_nodes.append(nodes)
        all_edges.append(edges)
        
        # Adjust edge indices by node offset
        senders = edge_index[:, 0] + node_offset
        receivers = edge_index[:, 1] + node_offset
        all_senders.append(senders)
        all_receivers.append(receivers)
        
        # Make edges bidirectional (as in algoperf)
        all_senders.append(receivers)
        all_receivers.append(senders)
        all_edges.append(edges)  # Duplicate edge features
        
        all_labels.append(item['labels'])
        all_weights.append(item['weights'])
        
        n_nodes.append(len(nodes))
        n_edges.append(len(edges) * 2)  # Bidirectional
        
        node_offset += len(nodes)
    
    # Concatenate all components
    graph = GraphsTuple(
        nodes=torch.cat(all_nodes, dim=0),
        edges=torch.cat(all_edges, dim=0),
        receivers=torch.cat(all_receivers, dim=0),
        senders=torch.cat(all_senders, dim=0),
        globals=None,  # Will be set in model
        n_node=torch.tensor(n_nodes, dtype=torch.long),
        n_edge=torch.tensor(n_edges, dtype=torch.long)
    )
    
    labels = torch.stack(all_labels)
    weights = torch.stack(all_weights)
    
    return graph, labels, weights


def binary_cross_entropy_with_mask(labels, logits, mask, label_smoothing=0.0):
    """Binary cross entropy loss with mask (following algoperf implementation)"""
    if not (logits.shape == labels.shape == mask.shape):
        raise ValueError(
            f'Shape mismatch between logits ({logits.shape}), targets '
            f'({labels.shape}), and weights ({mask.shape}).')
    if len(logits.shape) != 2:
        raise ValueError(f'Rank of logits ({logits.shape}) must be 2.')
    
    # To prevent propagation of NaNs during grad()
    # We mask over the loss for invalid targets later
    labels = torch.where(mask.to(torch.bool), labels, -1)
    
    # Apply label_smoothing
    num_classes = labels.shape[-1]
    smoothed_labels = ((1.0 - label_smoothing) * labels +
                       label_smoothing / num_classes)
    
    # Numerically stable implementation of BCE loss
    # This mimics TensorFlow's tf.nn.sigmoid_cross_entropy_with_logits()
    positive_logits = logits >= 0
    relu_logits = torch.where(positive_logits, logits, 0)
    abs_logits = torch.where(positive_logits, logits, -logits)
    losses = relu_logits - (logits * smoothed_labels) + (
        torch.log(1 + torch.exp(-abs_logits)))
    
    return torch.where(mask.to(torch.bool), losses, 0.)

@app.command()
def main(
    method: List[str] = typer.Option(["qr"], help="Eigenvector method to use (for SOAP)"),
    dtype: List[str] = typer.Option(["float32"], help="Data type to use"),
    latent_dim: int = 256,
    hidden_dims: Tuple[int] = (256,),
    num_message_passing_steps: int = 5,
    batch: int = 32,
    steps: int = 100,
    weight_decay: float = 0,
    opt: List[str] = typer.Option(["ForeachSOAP"], help="Optimizers to use"),
    win_condition_multiplier: float = 1.0,
    trials: int = 10,
    data_path: str = "/mnt/storage01/home/tomjodrell/ogbg",
    activation_fn_name: str = "relu",
    dropout_rate: float = 0.1,
    num_outputs: int = 128,
):
    """
    OGBG-MolPCBA benchmark using algoperf GNN implementation.
    Compatible with Heavyball repository format.
    """
    dtype = [getattr(torch, d) for d in dtype]
    
    # Create model using algoperf GNN
    model = GNN(
        num_outputs=num_outputs,
        dropout_rate=dropout_rate,
        activation_fn_name=activation_fn_name,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        num_message_passing_steps=num_message_passing_steps
    ).cuda()
    
    # Print model parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Load dataset - automatically downloads via TFDS
    dataset = OGBGDataset(data_dir=data_path, split='train')
    test_dataset = OGBGDataset(data_dir=data_path, split='test')
    
    # Create data loader
    train_loader = DataLoader(
        dataset,
        batch_size=batch,
        shuffle=True,
        collate_fn=collate_graphs,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues with TFDS
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch,
        shuffle=False,
        collate_fn=collate_graphs,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues with TFDS
        pin_memory=True
    )
    
    # Create data iterator that matches heavyball format
    data_iter = iter(train_loader)
    
    def data():
        nonlocal data_iter
        try:
            graph, labels, weights = next(data_iter)
        except StopIteration:
            # Reset iterator when exhausted
            data_iter = iter(train_loader)
            graph, labels, weights = next(data_iter)
        
        # Move to GPU
        graph = GraphsTuple(
            nodes=graph.nodes.cuda(),
            edges=graph.edges.cuda(),
            receivers=graph.receivers.cuda(),
            senders=graph.senders.cuda(),
            globals=graph.globals,
            n_node=graph.n_node.cuda(),
            n_edge=graph.n_edge.cuda()
        )
        
        return graph, (labels.cuda(), weights.cuda())
    
    # Multi-label binary classification loss with masking
    def loss_fn(output, target_tuple):
        labels, weights = target_tuple
        
        # Compute per-example losses
        per_example_losses = binary_cross_entropy_with_mask(
            labels, output, weights, label_smoothing=0.0
        )
        
        # Sum over tasks, mean over batch
        summed_losses = per_example_losses.sum(dim=-1)  # Sum over tasks
        valid_examples = weights.sum(dim=-1) > 0  # Examples with at least one valid label
        
        if valid_examples.sum() > 0:
            return summed_losses[valid_examples].mean()
        else:
            return summed_losses.mean()  # Fallback if no valid examples
    
    # Target loss based on algoperf OGBG benchmark (AP score, lower is better for loss)
    # We use 1 - AP as loss, so target loss = 1 - 0.280981 = 0.719019
    target_loss = win_condition_multiplier * 0.719019
    
    trial(
        model,
        data,
        loss_fn,
        loss_win_condition(target_loss),
        steps,
        opt[0],
        dtype[0],
        latent_dim,  # features parameter
        batch,
        weight_decay,
        method[0],
        num_message_passing_steps,  # sequence parameter
        num_outputs,  # some other parameter
        failure_threshold=10,
        base_lr=1e-3,
        trials=trials,
        estimate_condition_number=False,
        test_loader=None,
        track_variance=True,
    )

if __name__ == "__main__":
    app()

# steps per epoch: int(350343/32)