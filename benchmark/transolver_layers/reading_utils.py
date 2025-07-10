import json
from typing import Any
import h5py
import torch
import numpy as np
from pathlib import Path
from torch_geometric.data import Data
from src.dataset.connectivity_utils import compute_connectivity
import matplotlib.pyplot as plt

def read_h5_file(file_path: Path) -> dict[str, np.ndarray]:
    """ Reads a single timestep HDF5 file and extracts particle data.

    Args:
        file_path (Path): Path to the HDF5 file containing particle data.

    Returns:
        dict[str, np.ndarray]: A dictionary containing:
            - 'position': Particle positions of shape [N, 2]
            - 'velocity': Particle velocities of shape [N, 2]
            - 'acceleration': Particle accelerations of shape [N, 2]
            - 'particle_type': Particle types of shape [N]
    """
    with h5py.File(file_path, "r") as f:
        data = {
            'pressure': f['surface.pressure'][:],  # [N, 2]
            'verts': f['surface.verts'][:],  # [N, 2]
            'verts_normals': f['surface.verts_normals'][:],  # [N, 2]
        }

    return data


def prepare_torch_dataset(
    sequence_dict: dict[str, np.ndarray],
    metadata: dict[str, Any],
    C: int,
    filenames: list[Path],
    idx: int,
    use_relative_distance: bool = False,
    add_noise: bool = False,
    noise_std: float = 6.7e-4
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:

    # Current state
    pressure = torch.tensor(sequence_dict['pressure'], dtype=torch.float32)      # [N, 2]
    verts = torch.tensor(sequence_dict['verts'], dtype=torch.float32)      # [N, 2]
    verts_normals = torch.tensor(sequence_dict['verts_normals'], dtype=torch.float32)
    
    # Convert to torch tensors
    tensor_dict = {
        'pressure': pressure,           # [N, 2] or [N, 4] if using relative distance
        'verts': verts,           # [N, 2]
        'verts_normals': verts_normals,    # [N, 1]
    }

    model_input = torch.cat([
        tensor_dict['verts'], # [N, 2] or [N,4] if using relative distance
        tensor_dict['verts_normals'],           # [N, 2]
       
    ], dim=-1)

    return model_input, pressure

def create_dataset(
    file_path: str,
    metadata: dict[str, Any],
    C: int, 
    filenames: list[Path],
    idx: int,
    use_relative_distance: bool = False,
    add_noise: bool = False,
    noise_std: float = 6.7e-4
) -> tuple[torch.Tensor, torch.Tensor]:
    data = read_h5_file(file_path)
    model_input, target_acceleration = prepare_torch_dataset(data, metadata, C, filenames, idx, use_relative_distance, add_noise, noise_std)

    return model_input, target_acceleration 


def compute_edge_features(position: torch.Tensor, edge_index: torch.Tensor, connectivity_radius: float) -> torch.Tensor:
    """
    Computes edge features: relative displacement and normalized distance.

    Args:
        position: Tensor of shape [N, 2] containing particle positions.
        edge_index: Tensor of shape [2, num_edges] with sender and receiver indices.
        connectivity_radius: Float radius for normalization.

    Returns:
        edge_attr: Tensor of shape [num_edges, 3], containing:
            [dx, dy, normalized_distance]
    """
    senders = edge_index[0]  # [num_edges]
    receivers = edge_index[1]  # [num_edges]

    # Get positions of sender and receiver particles
    sender_pos = position[senders]     # [num_edges, 2]
    receiver_pos = position[receivers] # [num_edges, 2]

    # Compute relative displacement
    rel_disp = (sender_pos - receiver_pos) / connectivity_radius  # [num_edges, 2]

    # Compute normalized distance 
    rel_dist = torch.norm(rel_disp, dim=-1, keepdim=True)         # [num_edges, 1]

    # Concatenate to form edge features
    edge_features = torch.cat([rel_disp, rel_dist], dim=-1)           # [num_edges, 3]

    return edge_features


def create_pyg_dataset(
    file_path: str,
    metadata: dict[str, Any],
    C: int,
    filenames: list[Path],
    idx: int,
    use_relative_distance: bool = False,
    add_noise: bool = False,
    noise_std: float = 6.7e-4,
    add_self_edges: bool = True
) -> Data:
    """ Create a PyTorch Geometric Data object from a single timestep HDF5 file.

    Args:
        file_path (str): Path to the HDF5 file containing particle data.
        metadata (dict[str, Any]): Metadata containing simulation parameters and bounds.
        C (int): Number of previous velocity frames to include in the input.
        filenames (list[Path]): List of all filenames in the simulation folder.
        idx (int): Index of the current timestep in the filenames list.
        use_relative_distance (bool, optional): If True, uses relative distances to boundaries instead of absolute positions. Defaults to False.
        add_noise (bool, optional): If True, adds random walk noise to the velocity and position. Defaults to False.
        noise_std (float, optional): Standard deviation of the noise to be added. Defaults to 6.7e-4.
        add_self_edges (bool, optional): If True, includes self-loops in the graph connectivity. Defaults to True.

    Returns:
        Data: A PyTorch Geometric Data object containing:
            - x: Node features of shape [N, F] where N is the number of particles and F is the feature dimension.
            - edge_index: Graph connectivity in COO format of shape [2, E] where E is the number of edges.
            - edge_attr: Edge features of shape [E, D] where D is the feature dimension for edges.
            - y: Target acceleration of shape [N, 2] where N is the number of particles.    
    """

    # Read HDF5 and prepare features
    data = read_h5_file(file_path)

    node_features, target_acceleration = prepare_torch_dataset(
        sequence_dict=data,
        metadata=metadata,
        C=C,
        filenames=filenames,
        idx=idx,
        use_relative_distance=use_relative_distance,
        add_noise=add_noise,
        noise_std=noise_std
    )   
    # Get node positions for graph connectivity
    position = torch.tensor(data["position"], dtype=torch.float32)

    # Compute connectivity
    edge_index = compute_connectivity(
        positions=position,
        radius=metadata["default_connectivity_radius"],
        add_self_edges=add_self_edges
    )

    edge_features = compute_edge_features(
        position=position,
        edge_index=edge_index,
        connectivity_radius=metadata["default_connectivity_radius"]
    )

    return Data(
        x = node_features,         
        edge_index = edge_index,   
        edge_attr = edge_features, 
        y = target_acceleration    

    )

# if __name__ == "__main__":
#     FILE_PATH = "./datasets/Sand"
#     METADATA_PATH = "./datasets/Sand/metadata.json"
#     with open(METADATA_PATH, "r") as f:
#         metadata = json.load(f)
#     input, target = create_dataset(
#         file_path=f"{FILE_PATH}/test/test_0000/0090.h5",
#         metadata=metadata,
#         C = 5,
#         filenames = sorted(Path(f"{FILE_PATH}/test/test_0000").glob("*.h5")),
#         idx=90,
#         use_relative_distance = False,
#         add_noise = True,
#         noise_std = 6.7e-4
#     )

#     print("Input shape:", input.shape)   # Expect [N, 15] or [N, 17] if using relative distance
#     print("Target shape:", target.shape) # Expect [N, 2]
#     print("Sample input:", input[:5])    # First 5 particles
#     print("Sample target:", target[:5])

# import json
# from pathlib import Path
# from src.dataset.reading_utils import create_pyg_dataset
# import torch

if __name__ == "__main__":
    FILE_PATH = "./datasets/Sand"
    METADATA_PATH = f"{FILE_PATH}/metadata.json"
    
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)

    idx = 320
    filename = sorted(Path(f"{FILE_PATH}/train/train_0010").glob("*.h5"))[idx]

    data = create_pyg_dataset(
        file_path=filename,
        metadata=metadata,
        C=5,
        filenames=sorted(Path(f"{FILE_PATH}/train/train_0010").glob("*.h5")),
        idx=idx,
        use_relative_distance=False,
        add_noise=False,
        noise_std=6.7e-4,
        add_self_edges=False
    )

    print("Node feature shape:", data.x.shape)
    print("Target shape (y):", data.y.shape)
    print("Edge index shape:", data.edge_index.shape)
    print("Edge features shape:", data.edge_attr.shape)

    # Validate edge index
    max_index = data.x.shape[0] - 1
    assert data.edge_index.max() <= max_index, "Edge index refers to non-existent node"
    assert data.edge_index.min() >= 0, "Edge index has negative values"

    # Compute distances between connected nodes
    sender_pos = data.x[data.edge_index[0], :2]
    receiver_pos = data.x[data.edge_index[1], :2]
    rel_dist = torch.norm(sender_pos - receiver_pos, dim=-1)

    print(f"Edge distances (min/max): {rel_dist.min():.4f} / {rel_dist.max():.4f}")
    assert (rel_dist <= metadata["default_connectivity_radius"] * 1.01).all(), "Some edges exceed connectivity radius"

    mean_neighbors = (data.edge_index[1].bincount()).float().mean().item()
    print(f"\nAverage neighbors per node: {mean_neighbors:.2f}")

    # Spot check a few connections
    print("\nFirst 5 edges:")
    for i in range(min(5, data.edge_index.shape[1])):
        sender = data.edge_index[0, i].item()
        receiver = data.edge_index[1, i].item()
        print(f"Edge {i}: Sender {sender}, Receiver {receiver}")
        print(f"  Δx = {sender_pos[i].numpy() - receiver_pos[i].numpy()}, Distance = {rel_dist[i].item():.4f}")


    central_node = 1500
    neighbors = data.edge_index[0][data.edge_index[1] == central_node].unique()
    print(f"Node {central_node} has {len(neighbors)} neighbors: {neighbors.tolist()}")

    pos = data.x[:, :2]

    from sklearn.neighbors import NearestNeighbors

    # Convert positions to numpy for sklearn
    positions_np = pos.numpy()

    # Use sklearn to find neighbors within the same radius
    nn = NearestNeighbors(radius=metadata["default_connectivity_radius"])
    nn.fit(positions_np)
    scikit_neighbors = nn.radius_neighbors([positions_np[central_node]], return_distance=False)[0]

    print(f"[scikit-learn] Node {central_node} has {len(scikit_neighbors)} neighbors: {scikit_neighbors.tolist()}")


    # Compute zoom window
    # all_x = torch.cat([pos[central_node:central_node+1, 0], pos[neighbors, 0]])
    # all_y = torch.cat([pos[central_node:central_node+1, 1], pos[neighbors, 1]])
    # margin = 0.01  # Zoom margin
    # x_min, x_max = all_x.min().item() - margin, all_x.max().item() + margin
    # y_min, y_max = all_y.min().item() - margin, all_y.max().item() + margin

    # Create plot
    plt.figure(figsize=(6, 6), dpi=300)
    plt.scatter(pos[:, 0], pos[:, 1], c='lightgray', alpha=0.5, s=10, label="All nodes")
    plt.scatter(pos[central_node, 0], pos[central_node, 1], c='red', s=30, label="Central node")
    plt.scatter(pos[neighbors, 0], pos[neighbors, 1], c='blue', s=20, label="Neighbors")
    plt.legend()
    plt.axis('equal')
    # plt.xlim(x_min, x_max)
    # plt.ylim(y_min, y_max)
    plt.title("Central Node and Its Neighbors", fontsize=10)

    # Faint grid lines
    plt.grid(True, linestyle='--', linewidth=0.3, alpha=0.4)

    plt.savefig("out/connectivity_debug_plot.png")
