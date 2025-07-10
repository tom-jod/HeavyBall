from torch.utils.data import Dataset
from pathlib import Path
from typing import Any, Literal
from torch.utils.data import Dataset
from src.dataset.reading_utils import create_dataset

 
class ParticleDataset(Dataset):
    def __init__(
        self, 
        folder_path: str,
        dataset_name: Literal["WaterRamps", "Sand"],
        metadata: dict[str, Any],
        fold: Literal["train", "test", "valid"] = "train",
        C: int = 5,
        use_relative_distance: bool = False,
        add_noise: bool = False,
        noise_std: float = 6.7e-4
        ):
        self.C = C
        self._metadata = metadata
        self._fold_path = Path(folder_path) / dataset_name / fold
        self.use_relative_distance = use_relative_distance
        self.add_noise = add_noise
        self.noise_std = noise_std

        
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        file_path = self.filenames[idx]
        
        sample_dict, target = create_dataset(
                file_path,
                self._metadata,
                self.C,
                self.filenames,
                idx,
                self.use_relative_distance,
                self.add_noise,
                self.noise_std
            )

        return sample_dict, target

