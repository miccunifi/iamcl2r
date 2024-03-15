import numpy as np
from continuum.datasets import InMemoryDataset
from collections import defaultdict as dd
from torch.utils.data.sampler import BatchSampler

import logging
logger = logging.getLogger('Data-Utils')


def subsample_dataset(dataset, img_per_class):
    """ utility function to slice a dataset images per class. 
        Used for Continuum dataset. 
        It returns a new Continuum dataset with less img per class.
    """
    data, targets, task_ids = dataset.get_data()
    new_data = []
    new_targets = []
    new_task_ids = [] if task_ids is not None else None
    for cl in np.unique(targets):
        cl_idx = np.where(targets == cl)[0]
        new_data.append(data[cl_idx[:img_per_class]])
        new_targets.append(targets[cl_idx[:img_per_class]])
        if task_ids is not None:
            new_task_ids.append(task_ids[cl_idx[:img_per_class]])
    new_data = np.concatenate(new_data)
    new_targets = np.concatenate(new_targets)
    if task_ids is not None:
        new_task_ids = np.concatenate(new_task_ids)
    return InMemoryDataset(
        new_data, new_targets, new_task_ids,
        data_type=dataset.data_type
    )


class BalancedBatchSampler(BatchSampler):
    def __init__(self, dataset, batch_size, n_classes, n_samples, seen_classes, rehearsal=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.seen_classes = seen_classes
        self.rehearsal = rehearsal
        self.n_batches = self.n_samples // self.batch_size # drop last
        if self.n_batches == 0:
            self.n_batches = 1
            self.size = self.n_samples if rehearsal == 0 else self.n_samples//2
        elif rehearsal == 0:
            self.size = self.batch_size
        else:
            self.size = self.batch_size//2
        self.index_dic = dd(list)
        self.indices = []
        self.seen_indices = []
        for index, y in enumerate(self.dataset._y):
            if y not in self.seen_classes:
                self.indices.append(index)
            else:
                self.seen_indices.append(index)

    def __iter__(self):
        for _ in range(self.n_batches):
            batch = []
            if self.rehearsal > 0:
                replace = True if len(self.seen_indices) <= self.size else False
                batch.extend(np.random.choice(self.seen_indices, size=self.size, replace=replace))
            replace = True if len(self.indices) <= self.size else False
            batch.extend(np.random.choice(self.indices, size=self.size, replace=replace))
            yield batch

    def __len__(self):
        return self.n_batches