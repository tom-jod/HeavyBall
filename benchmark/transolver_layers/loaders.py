import json
from typing import Literal
from torch.utils.data import DataLoader
from base import ParticleDataset
from sampler import SimulationBatchSampler


def create_train_val_dataloaders(
    folder_path: str = "/udl/dinozaur/car_cfd_lethe/02_processed_data/",
    dataset_name: str = "02_processed_data",
    batch_size: int = 64,  
    num_workers: int = 4,
    use_relative_distance: bool = False,
    add_noise: bool = True,
    noise_std: float = 6.7e-4
) -> DataLoader:
    """
    Create a PyTorch DataLoader from a TFRecord dataset.

    Args:
        dataset_path (str): Path to the TFRecord dataset.
        batch_size (int): Batch size for the DataLoader.
        num_workers (int): Number of worker threads for loading data.
        shuffle (bool): Whether to shuffle the dataset.

    Returns:
        DataLoader: A PyTorch DataLoader.
    """
    if dataset_name not in ["WaterRamps", "Sand"]:
        raise ValueError(
            f"Unsupported dataset name: {dataset_name}. Supported datasets are: WaterRamps, Sand."
        )

    dataset_path = "/udl/dinozaur/car_cfd_lethe/02_processed_data/"
    

    
    train_dataset = ParticleDataset(
        dataset_name=dataset_name,
        folder_path = folder_path,
        fold="train",
        use_relative_distance=use_relative_distance,
        add_noise=add_noise,
        noise_std = noise_std,
    )

    validation_dataset = ParticleDataset(
        dataset_name=dataset_name,
        folder_path= folder_path,
        fold="valid",
        use_relative_distance=use_relative_distance,
        add_noise=False,
        noise_std = noise_std
    )

    # Use SimulationBatchSampler
    train_batch_sampler = SimulationBatchSampler(train_dataset, batch_size=batch_size, shuffle=True)
    val_batch_sampler = SimulationBatchSampler(validation_dataset, batch_size=batch_size, shuffle=False)

    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler= train_batch_sampler,
        num_workers=num_workers,

    )

    validation_dataloader = DataLoader(
        validation_dataset,
        batch_sampler= val_batch_sampler,
        num_workers=num_workers,

    )
    
    return train_dataloader, validation_dataloader




 