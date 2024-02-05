import torch
from torch.utils.data import DataLoader
from dataset.jsondataset import JsonDataset
from .dist import get_rank, get_world_size

def create_dataloader(config):
    dataset_train = JsonDataset(config.train_file, config=config)
    print("distributed debug message, RANK: {} and WORLD_SIZE: {}".format(get_rank(), get_world_size()))
    sampler = torch.utils.data.DistributedSampler(dataset_train, shuffle=True, num_replicas=get_world_size(), rank=get_rank()) if config.distributed else None

    loader_train = DataLoader(
        dataset_train,
        batch_size=config.batch_size,
        shuffle=(sampler is None),
        num_workers=config.workers,
        pin_memory=True,
        drop_last=True,
        sampler=sampler)
    
    if get_rank() == 0:
        total_image = len(dataset_train)
        print("Total training images: ", total_image)

    return dataset_train, loader_train