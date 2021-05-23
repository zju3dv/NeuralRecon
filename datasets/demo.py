import os
import numpy as np
import pickle
import copy
from PIL import Image
from torch.utils.data import Dataset


class DemoDataset(Dataset):
    def __init__(self, datapath, mode, transforms, nviews, n_scales):
        super(DemoDataset, self).__init__()
        self.datapath = datapath
        self.mode = mode
        self.n_views = nviews
        self.transforms = transforms

        assert self.mode in ["train", "val", "test"]
        self.metas = self.build_list()

        self.n_scales = n_scales
        self.epoch = None
        self.tsdf_cashe = {}
        self.max_cashe = 100

    def build_list(self):
        with open(os.path.join(self.datapath, 'fragments.pkl'), 'rb') as f:
            metas = pickle.load(f)

        return metas

    def __len__(self):
        return len(self.metas)

    def read_img(self, filepath):
        img = Image.open(filepath)
        return img

    def __getitem__(self, idx):
        meta = self.metas[idx]

        imgs = []
        intrinsics_list = meta['intrinsics']
        extrinsics_list = meta['extrinsics']
        intrinsics = np.stack(intrinsics_list)
        extrinsics = np.stack(extrinsics_list)

        for i, vid in enumerate(meta['image_ids']):
            # load images
            imgs.append(
                self.read_img(
                    os.path.join(self.datapath, 'images', '{}.jpg'.format(vid))))

        items = {
            'imgs': imgs,
            'intrinsics': intrinsics,
            'extrinsics': extrinsics,
            'scene': meta['scene'],
            'fragment': meta['scene'] + '_' + str(meta['fragment_id']),
            'epoch': [self.epoch],
            'vol_origin': np.array([0, 0, 0])
        }

        if self.transforms is not None:
            items = self.transforms(items)
        return items
