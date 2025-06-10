import torch
from torch.utils.data import DataLoader, Dataset

from src.systems.distributions import Sampleable


class DummyDataloader:
    """
    Dummy dataloader that does not load any data.
    """

    def __init__(self, steps_per_epoch: int):
        self.steps_per_epoch = steps_per_epoch
        self.current_step = 0

    def __iter__(self):
        self.current_step = 0
        return self

    def __len__(self):
        return self.steps_per_epoch

    def __next__(self):
        if self.current_step >= self.steps_per_epoch:
            raise StopIteration
        self.current_step += 1
        return torch.zeros(2)


class SampleableDataset(Dataset):
    """
    Dataset wrapper around a Sampleable object.
    """

    def __init__(self, sampleable: Sampleable, num_samples: int):
        self.sampleable = sampleable
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.sampleable.sample(1).squeeze(0)
