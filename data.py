import numpy as np
from torch.utils.data import Dataset, DataLoader
from monai import transforms
import json
import os


def read_json(datalist):
    with open(datalist) as f:
        json_data = json.load(f)

    return json_data


class BraTs2023_2D(Dataset):
    def __init__(self, data_list, phase='train'):
        super(BraTs2023_2D, self).__init__()
        self.data_list = data_list[phase]

        self.train_transform = transforms.Compose(
            [
                transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                transforms.RandScaleIntensityd(keys="image", factors=(-0.3, 0.3), prob=0.15),
                transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.1),
                transforms.RandGaussianNoised(keys="image", prob=0.15, mean=0.0, std=0.33),
                transforms.RandGaussianSmoothd(keys="image", sigma_x=(0.5, 1.5), sigma_y=(0.5, 1.5), sigma_z=(0.5, 1.5), prob=0.15),
                transforms.ToTensord(keys=["image", 'label']),
            ]
        )
        
        self.val_transform = transforms.Compose(
            [
                transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                transforms.ToTensord(keys=["image", 'label']),
            ]
        )

        self.test_transform = transforms.Compose(
            [
                transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                transforms.ToTensord(keys=["image", 'label']),
            ]
        )

        self.phase = phase

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        data = self.load_image(self.data_list[item])
        if self.phase == 'train':
            data = self.train_transform(data)
        elif self.phase == 'val':
            data = self.val_transform(data)
        elif self.phase == 'test':
            data = self.test_transform(data)

        return data

    def load_image(self, file_dic):
        data = np.load(file_dic['path'])
        
        t1c = data['t1c']
        t1n = data['t1n']
        t2w = data['t2w']
        t2f = data['t2f']
        
        image = np.stack([t1c, t1n, t2w, t2f], axis=0).reshape(4, 256, 256)
        
        label = data['label'] 

        return {
            'image': image,
            'label': label,
            'path': os.path.split(file_dic['path'])[-1]
        }
        
def get_loader(datalist_json,
               batch_size,
               num_works,
               phase=None):

    files = BraTs2023_2D(datalist=datalist_json)

    datasets = eval(datasets)(data_list=files, phase=phase)
    
    if phase != 'train':
        dataloader = DataLoader(datasets,
                                batch_size=batch_size,
                                num_workers=num_works,
                                pin_memory=True,
                                shuffle=False,
                                drop_last=True)
    else:
        dataloader = DataLoader(datasets,
                                batch_size=batch_size,
                                num_workers=num_works,
                                pin_memory=True,
                                shuffle=True,
                                drop_last=True)
    return dataloader
    
