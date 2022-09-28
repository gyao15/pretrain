from distutils.log import info
import mmcv
import numpy as np
import pickle
import copy
from PIL import Image
import tempfile
from os import path as osp
from .once_toolkits import Octopus
import torch
from torch.utils.data import Dataset
from .pipelines import Compose
from torchvision.transforms import RandomResizedCrop

from mmdet.datasets import DATASETS

@DATASETS.register_module()
class OnceImageDataset(Dataset):
    CLASSES = None
    def __init__(self, data_root, ann_file, pipeline=None, test_mode=False) -> None:
        super().__init__()
        self.data_root = data_root
        self.ann_file = ann_file
        self.test_mode = test_mode
        self.infos = mmcv.load(ann_file)
        self.transform = RandomResizedCrop(224)
        self.CLASSES = None

        if pipeline is not None:
            self.pipeline = Compose(pipeline)

        if not self.test_mode:
            self._set_group_flag()

    def __len__(self):
        return len(self.infos)

    def __getitem__(self, index):
        info_dict = self.infos[index]
        info_dict['img_fields'] = None
        info_dict['img_prefix'] = None
        example = self.pipeline(info_dict)
        example['img']._data = self.transform(example['img'].data)
        # print(example.keys())
        return example

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0. In 3D datasets, they are all the same, thus are all
        zeros.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        

        