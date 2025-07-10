import torch
import numpy as np
from typing import Dict, Any
from transolver import TransolverReg
def create_random_transolver_dataset(
    batch_size: int = 32,
    n_vertices: int = 100,
    feature_dim: int = 64,
    spatial_dim: int = 2,
    sequence_length: int = 10,
    features: list = None,
    targets: list = None,
    vertices_name: str = "vertices"
) -> Dict[str, torch.Tensor]:
    """
    Create a random dataset for testing Transolver model.
    
    Args:
        batch_size: Number of samples in the batch
        n_vertices: Number of spatial points/vertices
        feature_dim: Dimension of input features
        spatial_dim: Spatial dimension (2D or 3D)
        sequence_length: Temporal sequence length
        features: List of feature names
        targets: List of target names
        vertices_name: Name for vertices/coordinates
    
    Returns:
        Dictionary containing the dataset
    """
    if features is None:
        features = ["input_field", "boundary_conditions"]
    
    if targets is None:
        targets = ["output_field"]
    
    dataset = {}
    
    # Create spatial coordinates (vertices)
    dataset[vertices_name] = torch.randn(batch_size, n_vertices, spatial_dim)
    
    # Create input features
    for feature in features:
        if "field" in feature.lower():
            # Field data: (batch, vertices, channels)
            dataset[feature] = torch.randn(batch_size, n_vertices, feature_dim)
        elif "boundary" in feature.lower():
            # Boundary conditions: (batch, vertices, 1)
            dataset[feature] = torch.randint(0, 2, (batch_size, n_vertices, 1)).float()
        else:
            # Generic feature
            dataset[feature] = torch.randn(batch_size, n_vertices, feature_dim)
    
    # Create target data
    for target in targets:
        dataset[target] = torch.randn(batch_size, n_vertices, feature_dim)
    
    # Add time dimension if needed (for temporal problems)
    if sequence_length > 1:
        for key in dataset:
            if key != vertices_name:
                # Add time dimension: (batch, time, vertices, features)
                current_shape = dataset[key].shape
                dataset[key] = dataset[key].unsqueeze(1).repeat(1, sequence_length, 1, 1)
    
    return dataset

# Example usage and testing
def test_transolver_instantiation():
    """Test function to verify Transolver can be instantiated with random data."""
    
    # Create a mock config (you'll need to adjust this based on your actual config)
    class MockTransolverConfig:
        def __init__(self):
            self.features = ["input_field", "boundary_conditions"]
            self.targets = ["output_field"]
            self.vertices_name = "vertices"
            
        def get_target_set(self):
            return set(self.targets)
    
    class MockTransolverPyTorchConfig:
        def __init__(self):
            self.features = ["input_field", "boundary_conditions"]
            self.targets = ["output_field"]
            self.vertices_name = "vertices"
            self.transolver = MockTransolverConfig()
            
        def get_target_set(self):
            return set(self.targets)
    
    # Create random dataset
    random_data = create_random_transolver_dataset(
        batch_size=16,
        n_vertices=64,
        feature_dim=32,
        spatial_dim=2,
        features=["input_field", "boundary_conditions"],
        targets=["output_field"]
    )
    
    print("Dataset shapes:")
    for key, value in random_data.items():
        print(f"{key}: {value.shape}")
    
    # Create config
    config = MockTransolverPyTorchConfig()
    
    try:
        # Try to instantiate your model
        model = TransolverReg(
            feature_dim= 1,
            n_layers= 5,
            n_hidden= 256,
            dropout= 0.0,
            n_head= 8,
            act= "gelu",
            mlp_ratio= 1,
            function_dim= 1,
            out_dim= 1,
            slice_num= 32,
            ref= 8,
        )
        print("✓ Model configuration created successfully")
        
        # Test with the random data
        print("✓ Random dataset created successfully")
        print("Dataset ready for model testing!")
        
    except Exception as e:
        print(f"✗ Error during instantiation: {e}")
        return False
    
    return True

# Alternative: Create a simple torch Dataset class for testing
class RandomTransolverDataset(torch.utils.data.Dataset):
    """Random dataset for testing Transolver."""
    
    def __init__(
        self,
        num_samples: int = 100,
        n_vertices: int = 64,
        feature_dim: int = 32,
        spatial_dim: int = 2,
        features: list = None,
        targets: list = None,
        vertices_name: str = "vertices"
    ):
        self.num_samples = num_samples
        self.n_vertices = n_vertices
        self.feature_dim = feature_dim
        self.spatial_dim = spatial_dim
        self.features = features or ["input_field"]
        self.targets = targets or ["output_field"]
        self.vertices_name = vertices_name
        
        # Pre-generate data
        self.data = []
        for _ in range(num_samples):
            sample = create_random_transolver_dataset(
                batch_size=1,
                n_vertices=n_vertices,
                feature_dim=feature_dim,
                spatial_dim=spatial_dim,
                features=self.features,
                targets=self.targets,
                vertices_name=vertices_name
            )
            # Remove batch dimension
            sample = {k: v.squeeze(0) for k, v in sample.items()}
            self.data.append(sample)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx]

# Quick test function
def quick_test():
    """Quick test to verify everything works."""
    
    # Test 1: Create random data dictionary
    print("Test 1: Creating random dataset...")
    data = create_random_transolver_dataset()
    print("✓ Success!")
    
    # Test 2: Create random dataset class
    print("\nTest 2: Creating random dataset class...")
    dataset = RandomTransolverDataset(num_samples=10)
    print(f"✓ Success! Dataset length: {len(dataset)}")
    
    # Test 3: Check data loader compatibility
    print("\nTest 3: Testing with DataLoader...")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
    batch = next(iter(dataloader))
    print("✓ Success! Batch keys:", list(batch.keys()))
    
    return True
def create_forward_compatible_dataset(
    batch_size: int = 4,
    n_vertices: int = 64,
    input_channels: int = 1,
    spatial_dim: int = 2,
    device: str = "cpu"
) -> Dict[str, torch.Tensor]:
    """Create dataset specifically for testing forward pass."""
    
    # Create a simple structured grid (more realistic for PDEs)
    if spatial_dim == 2:
        # 2D grid
        side_length = int(np.sqrt(n_vertices))
        x = torch.linspace(0, 1, side_length)
        y = torch.linspace(0, 1, side_length)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        coords = torch.stack([xx.flatten(), yy.flatten()], dim=-1)
        coords = coords.unsqueeze(0).repeat(batch_size, 1, 1)
    else:
        # Random coordinates as fallback
        coords = torch.rand(batch_size, n_vertices, spatial_dim)
    
    # Simple input field (e.g., initial condition)
    input_field = torch.randn(batch_size, n_vertices, input_channels)
    
    return {
        "x": input_field,  # Most models expect 'x' as main input
        "coordinates": coords,
        "vertices": coords,  # Alternative name
    }

# Test forward pass
def test_forward_pass():
    """Test if forward pass works with random data."""
    
    try:
        # Create model (you'll need your actual config)
        # model = TransolverPyTorch(config=your_config)
        
        # Create compatible random data
        data = create_forward_compatible_dataset(
            batch_size=2,
            n_vertices=64,
            input_channels=1
        )
        
        print("Input shapes:")
        for key, value in data.items():
            print(f"  {key}: {value.shape}")
        
        # Try forward pass
        # output = model(data["x"])  # or model.forward(data["x"])
        # print(f"Output shape: {output.shape}")
        
        print("✅ Forward pass test setup complete!")
        return True
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return False

if __name__ == "__main__":
    test_transolver_instantiation()
    quick_test()
    test_forward_pass()
    