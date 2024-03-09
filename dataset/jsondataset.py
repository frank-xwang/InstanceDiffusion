import os
import json
import torch
from torch.utils.data import Dataset
from dataset.decode_item import decode

class JsonDataset(Dataset):
    """Custom dataset."""
    def __init__(self, text_file, root_dir="xudongw/DATASETS/", decode_func=None, config=None):
        """
        Arguments:
            text_file (string): path to the text file with paths to all json files.
            root_dir (string): home directory.
            decode_func (callable): decode function to decode json file.
        """
        with open(os.path.join(root_dir, text_file), 'r') as f:
            lines = f.readlines()
            self.train_files = [line.strip() for line in lines]
        self.root_dir = root_dir
        if decode_func is None:
            yaml_params = config.train_dataset_names['Grounding']
            if yaml_params is not None:
                params = yaml_params
            else:
                params = {}
            if config.count_dup:
                params['count_dups_make_a_sentence'] = True
            params['random_blip'] = config.random_blip
            params['return_att_masks'] = config.use_masked_att
            params['add_inst_cap_2_global'] = config.add_inst_cap_2_global

            self.decode_func = decode(**params)
        else:
            self.decode_func = decode_func

    def __len__(self):
        return len(self.train_files)

    def __getitem__(self, idx):
        # Opening JSON file
        f = open(os.path.join(self.root_dir, self.train_files[idx]))
        try:
            data = json.load(f)
            data = self.decode_func(data)
            f.close()
        except:
            # show error message and return None
            assert 1 == 0, os.path.join(self.root_dir, self.train_files[idx])
        return data


def sub_batch(batch, num=1):
    # choose first num samples in given batch 
    num = num if num > 1 else 1 
    for k in batch:
        batch[k] = batch[k][0:num]
    return batch

def batch_to_device(batch, device):
    for k in batch:
        if isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].to(device)
        if isinstance(batch[k], list):
            for i in range(len(batch[k])): # check all dicts in list
                if isinstance(batch[k][i], dict):
                    for j in batch[k][i]: # check all keys in dict
                        if isinstance(batch[k][i][j], torch.Tensor):
                            batch[k][i][j] = batch[k][i][j].to(device)
    return batch