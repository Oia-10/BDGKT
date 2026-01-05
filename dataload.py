import os
from torch.utils.data import Dataset, DataLoader
import dgl
import torch


def load_data(data_path):
    data_dir = []
    dir_list = os.listdir(data_path)
    dir_list.sort()
    for filename in dir_list:
        for fil in os.listdir(os.path.join(data_path, filename)):
            data_dir.append(os.path.join(os.path.join(data_path, filename), fil))
    return data_dir

class myFloder(Dataset):
    def __init__(self, root_dir, loader):
        self.root = root_dir
        self.loader = loader
        self.dir_list = load_data(root_dir)
        self.size = len(self.dir_list)

    def __getitem__(self, index):
        dir_ = self.dir_list[index]
        data = self.loader(dir_)
        return data

    def __len__(self):
        return self.size
    
def collate(data):
    graph = []

    target_user = []
    target_item = []
    target_res = []

    user_alis = []
    item_alis = []
    item_exist = []
    
    
    for da in data:
        graph.append(da[0][0])

        target_user.append(da[1]['user'])
        target_item.append(da[1]['target_item'])
        target_res.append(da[1]['target_res'])
        user_alis.append(da[1]['u_alis'])
        item_alis.append(da[1]['target_alis'])
        item_exist.append(da[1]['target_item_exist'])
      
    return dgl.batch(graph), \
           torch.Tensor(target_user).long(), torch.Tensor(target_item).long(), torch.Tensor(target_res).long(),\
           torch.Tensor(user_alis).long(), torch.Tensor(item_alis).long(), torch.Tensor(item_exist).long()