from torch.utils.data import DataLoader
import random
import numpy as np

class BalancedBatchSampler_Up:
    def __init__(self, labels, batch_size):
        """
        Balanced batch sampler to enforce equal representation of classes in each batch.
        
        Args:
            labels (list or array): List of labels corresponding to each sample in the dataset.
            batch_size (int): Total batch size. Must be divisible by the number of classes.
        """
        self.labels = np.array(labels)
        self.classes = np.unique(self.labels)
        self.batch_size = batch_size
        self.batch_size_per_class = batch_size // len(self.classes)
        assert batch_size % len(self.classes) == 0, "Batch size must be divisible by the number of classes."
        
        self.class_indices = {cls: np.where(self.labels == cls)[0].tolist() for cls in self.classes}
        self.max_samples = max(len(indices) for indices in self.class_indices.values())

    def __iter__(self):
        """
        Yields balanced batches of indices.
        """
        shuffled_class_indices = {cls: random.sample(indices, len(indices)) for cls, indices in self.class_indices.items()}
        oversampled_class_indices = {cls: indices + random.choices(indices, k=self.max_samples - len(indices)) 
                                     for cls, indices in shuffled_class_indices.items()}

        batches = []
        for i in range(0, self.max_samples, self.batch_size_per_class):
            batch = []
            for cls, indices in oversampled_class_indices.items():
                batch.extend(indices[i:i + self.batch_size_per_class])
            random.shuffle(batch)
            if len(batch) == self.batch_size:  # Ensure full batch
                batches.append(batch)

        return iter(batches)

    def __len__(self):
        """
        Returns the number of batches.
        """
        return self.max_samples // self.batch_size_per_class

class BalancedBatchSampler_Down:
    def __init__(self, labels, batch_size):
        """
        Balanced batch sampler to enforce equal representation of classes in each batch using downsampling.
        
        Args:
            labels (list or array): List of labels corresponding to each sample in the dataset.
            batch_size (int): Total batch size. Must be divisible by the number of classes.
        """
        self.labels = np.array(labels)
        self.classes = np.unique(self.labels)
        self.batch_size = batch_size
        self.batch_size_per_class = batch_size // len(self.classes)
        assert batch_size % len(self.classes) == 0, "Batch size must be divisible by the number of classes."
        
        self.class_indices = {cls: np.where(self.labels == cls)[0].tolist() for cls in self.classes}
        self.min_samples = min(len(indices) for indices in self.class_indices.values())

    def __iter__(self):
        """
        Yields balanced batches of indices.
        """
        
        shuffled_class_indices = {cls: random.sample(indices, len(indices)) for cls, indices in self.class_indices.items()}
        
        downsampled_class_indices = {cls: indices[:self.min_samples] for cls, indices in shuffled_class_indices.items()}

        batches = []
        for i in range(0, self.min_samples, self.batch_size_per_class):
            batch = []
            for cls, indices in downsampled_class_indices.items():
                batch.extend(indices[i:i + self.batch_size_per_class])
            random.shuffle(batch)
            if len(batch) == self.batch_size:
                batches.append(batch)

        return iter(batches)

    def __len__(self):
        """
        Returns the number of batches.
        """
        return self.min_samples // self.batch_size_per_class
