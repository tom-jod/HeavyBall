from torch.utils.data import Sampler
import random

class SimulationBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, shuffle=True):
        """
        Args:
            dataset: The dataset (an instance of ParticleDataset)
            batch_size: Number of samples per batch (should match the number of samples per trajectory).
            shuffle: Whether to shuffle the list of trajectories.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.simulation_to_indices = self._group_indices_by_simulation()
        self.simulation_ids = list(self.simulation_to_indices.keys())

    def _group_indices_by_simulation(self):
        simulation_to_indices = {}
        for idx, path in enumerate(self.dataset.filenames):
            simulation_id = path.parent.name
            if simulation_id not in simulation_to_indices:
                simulation_to_indices[simulation_id] = []
            simulation_to_indices[simulation_id].append(idx)
        return simulation_to_indices

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.simulation_ids)

        for sim_id in self.simulation_ids:
            indices = self.simulation_to_indices[sim_id]
            if self.shuffle:
                random.shuffle(indices)  # Shuffle samples within a simulation

            # Yield in batches of batch_size
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i+self.batch_size]
                yield batch

    def __len__(self):
        # Total number of batches = sum over simulations
        total_batches = 0
        for indices in self.simulation_to_indices.values():
            total_batches += (len(indices) + self.batch_size - 1) // self.batch_size  # ceiling division
        return total_batches
