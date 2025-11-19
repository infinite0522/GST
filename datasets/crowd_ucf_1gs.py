import h5py
from PIL import Image
import torch.utils.data as data
import os
from glob import glob
import torch
import torchvision.transforms.functional as F
from torchvision import transforms
import random
import numpy as np


def random_crop(im_h, im_w, crop_h, crop_w):
    res_h = im_h - crop_h
    res_w = im_w - crop_w
    i = random.randint(0, res_h)
    j = random.randint(0, res_w)
    return i, j, crop_h, crop_w


def cal_innner_area(c_left, c_up, c_right, c_down, bbox):
    inner_left = np.maximum(c_left, bbox[:, 0])
    inner_up = np.maximum(c_up, bbox[:, 1])
    inner_right = np.minimum(c_right, bbox[:, 2])
    inner_down = np.minimum(c_down, bbox[:, 3])
    inner_area = np.maximum(inner_right - inner_left, 0.0) * np.maximum(inner_down - inner_up, 0.0)
    return inner_area


class Crowd(data.Dataset):
    def __init__(self, root_path, crop_size,
                 downsample_ratio, is_gray=False,
                 method='train', gs_path='gs_params'):
        self.root_path = root_path
        self.im_list = sorted(glob(os.path.join(self.root_path, '*.jpg')))
        if method not in ['train', 'val', 'test']:
            raise Exception("not implement")
        self.method = method

        self.c_size = crop_size
        self.d_ratio = downsample_ratio
        assert self.c_size % self.d_ratio == 0
        self.dc_size = self.c_size // self.d_ratio

        self.gs_path = gs_path
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@gs_path:", gs_path)

        if is_gray:
            self.trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        else:
            self.trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, item):
        img_path = self.im_list[item]
        gd_path = img_path.replace('jpg', 'npy')
        img = Image.open(img_path).convert('RGB')
        if self.method == 'train':
            keypoints = np.load(gd_path)
            gs_path = img_path.replace('.jpg', '_gs_params.h5').replace('IMG', self.gs_path + '/IMG')
            with h5py.File(gs_path, 'r') as f:
                gs_params = f['params'][:]
            if len(keypoints) > 0:
                gs_params = gs_params[:, :5]
                scales = gs_params[:, 2:4]
                rotations = gs_params[:, 4:5]
            else:
                scales = np.array([])
                rotations = np.array([])
                keypoints = keypoints.reshape((0,2))
                scales = scales.reshape((0,2))
                rotations = rotations.reshape((0,1))

            if np.isnan(scales).any():
                print(f"{img_path} contains NaN scale values")
                scales = np.repeat(keypoints[:, -1][:, np.newaxis], repeats=2, axis=1) / 6.0

            return self.train_transform(img, keypoints, scales, rotations)
        elif self.method == 'val':
            keypoints = np.load(gd_path)
            img = self.trans(img)
            name = os.path.basename(img_path).split('.')[0]
            return img, len(keypoints), name
        elif self.method == 'test':
            img = self.trans(img)
            name = os.path.basename(img_path).split('.')[0]
            name = name.split('_')[1]
            return img, name

    def train_transform(self, img, keypoints, scales, rotations):

        """random crop image patch and find people in it"""
        wd, ht = img.size
        # assert len(keypoints) > 0
        if random.random() > 0.88:
            img = img.convert('L').convert('RGB')
        re_size = random.random() * 0.5 + 0.75
        wdd = (int)(wd * re_size)
        htt = (int)(ht * re_size)
        if min(wdd, htt) >= self.c_size:
            wd = wdd
            ht = htt
            img = img.resize((wd, ht))
            keypoints = keypoints * re_size
            scales = scales * re_size
        st_size = min(wd, ht)
        assert st_size >= self.c_size
        i, j, h, w = random_crop(ht, wd, self.c_size, self.c_size)
        img = F.crop(img, i, j, h, w)
        if len(keypoints) > 0:
            nearest_dis = np.clip(keypoints[:, 2], 4.0, 128.0)
            points_left_up = keypoints[:, :2] - nearest_dis[:, None] / 2.0
            points_right_down = keypoints[:, :2] + nearest_dis[:, None] / 2.0
            bbox = np.concatenate((points_left_up, points_right_down), axis=1)
            inner_area = cal_innner_area(j, i, j+w, i+h, bbox)
            origin_area = nearest_dis * nearest_dis
            ratio = np.clip(1.0 * inner_area / origin_area, 0.0, 1.0)
            mask = (ratio >= 0.3)

            target = ratio[mask]
            keypoints = keypoints[mask]
            keypoints = keypoints[:, :2] - [j, i]
            scales, rotations = scales[mask], rotations[mask]

        if len(keypoints) > 0:
            if random.random() > 0.5:
                img = F.hflip(img)
                keypoints[:, 0] = w - keypoints[:, 0]
                rotations = -1 * rotations
        else:
            target = np.array([])
            if random.random() > 0.5:
                img = F.hflip(img)
        return (self.trans(img), torch.from_numpy(keypoints.copy()).float(),
                torch.from_numpy(scales.copy()).float(), torch.from_numpy(rotations.copy()).float(),
                torch.from_numpy(target.copy()).float(), st_size)