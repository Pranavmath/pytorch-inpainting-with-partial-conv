import random
import torch
from PIL import Image
from glob import glob
import os


class Places2(torch.utils.data.Dataset):
    def __init__(self, img_root, mask_root, img_transform, mask_transform,
                 split='train'):
        super(Places2, self).__init__()
        self.img_transform = img_transform
        self.mask_transform = mask_transform

        # use about 8M images in the challenge dataset
        if split == 'train':
            self.paths = glob('{:s}/*.jpg'.format(img_root),
                              recursive=True)
        else:
            self.paths = glob('{:s}/*'.format(img_root, split))

        self.img_root = img_root
        self.mask_root = mask_root

    def __getitem__(self, index):
        gt_img = Image.open(os.path.join(self.img_root, f"{index}.jpg"))
        gt_img = self.img_transform(gt_img.convert('RGB'))

        mask = Image.open(os.path.join(self.mask_root, f"{index}.jpg"))
        mask = self.mask_transform(mask.convert('RGB'))
        return gt_img * mask, mask, gt_img

    def __len__(self):
        return len(self.paths)
