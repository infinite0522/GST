import argparse
import csv
import pickle

import cv2
import yaml
from PIL import Image
import torch.utils.data as data
import os
from glob import glob
import torch
import torchvision.transforms.functional as F
from PIL.Image import Resampling
from matplotlib import pyplot as plt
from torchvision import transforms
import random
import numpy as np
from tqdm import tqdm
import h5py

from GS2D.image_fitting_bk2dgs_1gs import SimpleTrainer
from GS2D.give_required_data import coords_normalize, coords_reverse

def image_to_tensor(img: Image, width=None, height=None):
    import torchvision.transforms as transforms

    if width is not None and height is not None:
        transform = transforms.Compose([
            transforms.Resize([width, height]),
            transforms.ToTensor()
        ])
    else:
        transform = transforms.ToTensor()
    img_tensor = transform(img).permute(1, 2, 0)[..., :3]
    return img_tensor

class Counting_2dgs:
    def __init__(self, config_file_path, is_gray=False):
        with open(config_file_path, 'r') as config_file:
            config = yaml.safe_load(config_file)

        self.root_path = config["root_path"]
        self.im_list = sorted(glob(os.path.join(self.root_path, '*.jpg')))
        self.im_list = [img_path for img_path in self.im_list if 'gs' not in img_path]
        self.d_ratio = config["downsample_ratio"]
        self.num_epochs = config["num_epochs"]
        self.use_bk_mask = config["use_bk_mask"]
        self.learning_rate = config["learning_rate"]
        self.num_back_points = config["bk_num"]
        self.gt_points = config["gt_points"]    # True or False, whether gt_points exist


        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def batch_saving(self):
        output_csv_path = os.path.join(self.root_path, "gs_params/loss_results.csv")

        # Open a CSV file for writing the results
        with open(output_csv_path, mode='w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['File Name', 'Render_Loss', 'Shape_Loss'])

            for img_path in tqdm(self.im_list):

                gs_params_save_path = img_path.replace('.jpg', '_gs_params.h5').replace('IMG', 'gs_params/IMG')
                if os.path.exists(gs_params_save_path) and os.path.exists(out_img_save_path):
                    continue

                img = Image.open(img_path).convert('RGB')
                print(img.size)
                width, height = img.size
                if self.d_ratio != 1:
                    re_width, re_height = img.size[0] // self.d_ratio, img.size[1] // self.d_ratio
                    img.resize((re_width, re_height), Image.BILINEAR)
                    gt_img_tensor = image_to_tensor(img, width, height)
                else:
                    gt_img_tensor = image_to_tensor(img)

                gt_path = img_path.replace('jpg', 'npy')
                gt_points = torch.tensor(np.load(gt_path))

                if not len(gt_points) > 0:
                    params = np.array([])
                    with h5py.File(os.path.join(gs_params_save_path), 'w') as f:
                        f.create_dataset('params', data=params, compression='gzip')
                    continue

                if self.d_ratio != 1:
                    gt_points[:,:2] = coords_reverse(coords_normalize(gt_points[:,:2], [height, width]),[re_height, re_width], device=torch.device('cpu'))
                num_points = gt_points.shape[0] + self.num_back_points
                print(f"{img_path} processsing:")

                trainer = SimpleTrainer(gt_image=gt_img_tensor, num_points=num_points, gt_points=gt_points, bk_num=self.num_back_points, use_bk_mask=self.use_bk_mask)
                render_loss, shape_loss, _, means, scales, rotations, colors = trainer.train(
                    iterations=self.num_epochs,
                    lr=self.learning_rate,
                    save_imgs=False
                )

                params = torch.cat((means, scales, rotations, colors), dim=1)
                params = params.cpu().detach().numpy()


                with h5py.File(os.path.join(gs_params_save_path), 'w') as f:
                    f.create_dataset('params', data=params, compression='gzip')

                csv_writer.writerow([os.path.basename(img_path), render_loss.cpu().detach().numpy(), shape_loss.cpu().detach().numpy()])
                csvfile.flush()

if __name__ == '__main__':
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Run Counting with configuration file.')
    parser.add_argument('--config', type=str, help='Path to the config file', required=True)
    args = parser.parse_args()

    torch.cuda.empty_cache()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

    Counting_gs = Counting_2dgs(args.config)
    Counting_gs.batch_saving()
