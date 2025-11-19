import cv2
import math
import os
import time
#from pathlib import Path
from typing import Optional

import numpy as np
import torch
import tyro
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors
from torch import Tensor, optim, nn

from gsplat import rasterization, DefaultStrategy

from GS2D.give_required_data import get_colour
from .loss import d_shape_loss

def knn(x: Tensor, K: int = 4) -> Tensor:
    x_np = x.cpu().numpy()
    model = NearestNeighbors(n_neighbors=K, metric="euclidean").fit(x_np)
    distances, _ = model.kneighbors(x_np)
    return torch.from_numpy(distances).to(x)


class SimpleTrainer:
    """Trains random gaussians to fit an image."""

    def __init__(
            self,
            gt_image: Tensor,
            num_points: int = 2000,
            gt_points: Tensor = None,
            bk_num: int = 0,
            use_bk_mask: bool = False,
    ):
        self.device = torch.device("cuda:0")
        self.gt_image = gt_image.to(device=self.device)
        self.bk_num = bk_num
        if gt_points is None:
            self.num_points = num_points
        else:
            self.num_points = gt_points.shape[0]
        if bk_num > 0:
            self.num_points = self.num_points + bk_num
        self.use_bk_mask = use_bk_mask
        print(self.num_points)

        fov_x = math.pi / 2.0
        self.H, self.W = gt_image.shape[0], gt_image.shape[1]
        self.focal = 0.5 * float(self.W) / math.tan(0.5 * fov_x)
        self.img_size = torch.tensor([self.W, self.H, 1], device=self.device)

        self._init_gaussians(gt_points)

    def _init_gaussians(self, points: Tensor):
        """Random gaussians"""
        self.gt_means = points[:, :2].to(dtype=torch.float32, device=self.device)
        if self.bk_num > 0:
            self.bk_means = 2 * (torch.rand(self.bk_num, 2, dtype=torch.float32, device=self.device) - 0.5)
            self.bk_means[:, 0] = self.bk_means[:, 0] * self.W / 2 + self.W / 2
            self.bk_means[:, 1] = self.bk_means[:, 1] * self.H / 2 + self.H / 2
            self.means = torch.cat([self.gt_means, self.bk_means], dim=0)
        else:
            self.means = self.gt_means

        if len(points) > 1:
            k_neighbor = 4 if len(points) > 3 else len(points)
            dist2_avg = (knn(self.gt_means, k_neighbor)[:, 1:] ** 2).mean(dim=-1) # [N,]
        else:
            dist2_avg = torch.tensor([64.],dtype=torch.float32,device=self.device)

        dist_avg = torch.sqrt(dist2_avg).unsqueeze(-1)
        self.dis = dist_avg.repeat(1, 2) / 6.0
        self.dis = torch.clamp(self.dis, min=2.0, max=0.015 * max(self.W, self.H))

        if self.bk_num > 0:
            dist2_avg_all = (knn(self.means, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
            dist_avg_all = torch.sqrt(dist2_avg_all).unsqueeze(-1)
        else:
            dist_avg_all = dist_avg.clone()

        self.scales = torch.clamp(torch.log(dist_avg_all).repeat(1, 2), min=2.0)

        img_array = self.gt_image.clone()
        coords = self.means.clone().to(torch.int)
        coords[:, 0] = torch.clamp(coords[:, 0], min=0,max=self.H-1)
        coords[:, 1] = torch.clamp(coords[:, 1], min=0, max=self.W-1)

        print("######################################",self.gt_image.shape)
        eps = 1e-3
        input_tensor = torch.clamp(img_array[coords[:, 0], coords[:, 1]], eps, 1 - eps)
        self.rgbs = torch.logit(input_tensor).to(self.device)
        self.rotations = torch.logit(torch.rand(self.num_points, 1, device=self.device))
        self.opacities = torch.logit(torch.ones((self.num_points), device=self.device))

        # background generation
        if self.use_bk_mask:
            mask, _, meta = rasterization(
                self.gt_means,
                torch.zeros(self.num_points - self.bk_num, 1, device=self.device),
                self.dis,
                torch.ones((self.num_points - self.bk_num), device=self.device),
                torch.ones(self.num_points - self.bk_num, 3, device=self.device),
                # self.viewmat[None],
                # K[None],
                self.W,
                self.H,
                packed=False,
            )

            mask = (mask[0] == 0).int()
            self.background = self.gt_image * mask
            print(self.background.shape, self.background.dtype)
            # print(self.scales, meta['radii'])
        self.bk_ratio = self.H * self.W * 3 / torch.sum(1 - mask) if self.use_bk_mask else 1.
        print(f"self.bk_ratio: {self.bk_ratio}")

        self.gt_means.requires_grad = False
        if self.bk_num > 0:
            self.bk_means.requires_grad = True
        self.scales.requires_grad = True
        self.dis.requires_grad = False
        self.rotations.requires_grad = True
        self.rgbs.requires_grad = True
        self.opacities.requires_grad = False

    def train(
            self,
            iterations: int = 1000,
            lr: float = 0.01,
            save_imgs: bool = False,
    ):

        if self.bk_num > 0:
            optimizer = optim.Adam([
                {'params': self.gt_means, 'lr': lr},
                {'params': self.bk_means, 'lr': lr},
                {'params': self.rgbs, 'lr': lr},
                {'params': self.rotations, 'lr': lr},
                {'params': self.scales, 'lr': lr},
            ])
        else:
            optimizer = optim.Adam([
                {'params': self.gt_means, 'lr': lr},
                {'params': self.rgbs, 'lr': lr},
                {'params': self.rotations, 'lr': lr},
                {'params': self.scales, 'lr': lr},
            ])

        mse_loss = torch.nn.MSELoss()
        l1_loss = torch.nn.L1Loss()
        frames = []
        times = [0] * 2  # rasterization, backward
        for iter in range(iterations):
            start = time.time()
            if iter < iterations - 1:
                renders, _, _ = rasterization(
                    self.means,
                    torch.sigmoid(self.rotations) * torch.pi,
                    torch.relu(self.scales)+1.,
                    torch.sigmoid(self.opacities),
                    torch.sigmoid(self.rgbs),
                    # self.viewmat[None],
                    # K[None],
                    self.W,
                    self.H,
                    packed=False,
                )
            else:
                renders, _, meta = rasterization(
                    self.means,
                    torch.sigmoid(self.rotations) * torch.pi,
                    torch.relu(self.scales)+1.,
                    torch.sigmoid(self.opacities),
                    torch.sigmoid(self.rgbs),
                    # self.viewmat[None],
                    # K[None],
                    self.W,
                    self.H,
                    packed=False,
                )
            out_img = renders[0]
            torch.cuda.synchronize()
            times[0] += time.time() - start
            if self.use_bk_mask:
                out_img = out_img + self.background
            if iter < iterations - 20000:
                loss = l1_loss(out_img, self.gt_image) * self.bk_ratio
                if iter % 100 == 0:
                    print(f"Iteration {iter + 1}/{iterations}, Loss: {loss}")
            else:
                # self.means.requires_grad = True
                ll1_loss = l1_loss(out_img, self.gt_image) * self.bk_ratio
                shape_loss = d_shape_loss(self.scales[:(self.num_points - self.bk_num)], shape_threshold=1.5)
                loss = ll1_loss + 0.2 * shape_loss # add shape loss
                if iter % 1000 == 0:
                    print(f"Iteration {iter + 1}/{iterations}, Loss: {loss}")
                    print(f"render_loss: {ll1_loss}, shape_loss: {shape_loss}")
            optimizer.zero_grad()
            start = time.time()
            loss.backward()
            torch.cuda.synchronize()
            times[1] += time.time() - start
            optimizer.step()

            if save_imgs and iter % 5 == 0:
                out_img = torch.clamp(out_img, min=0.0, max=1.0)
                frames.append((out_img.detach().cpu().numpy() * 255.0).astype(np.uint8))

        if save_imgs:
            # save them as a gif with PIL
            frames = [Image.fromarray(frame) for frame in frames]
            out_dir = os.path.join(os.getcwd(), "examples/renders")
            print(out_dir)
            os.makedirs(out_dir, exist_ok=True)
            frames[0].save(
                f"{out_dir}/training.gif",
                save_all=True,
                append_images=frames[1:],
                optimize=False,
                duration=5,
                loop=0,
            )

            frames[0].save(
                f"{out_dir}/first_frame.png",
                format="PNG"
            )
            frames[-1].save(
                f"{out_dir}/last_frame.png",
                format="PNG"
            )

        print(f"Information:\nResolution: {self.W} * {self.H}, num_points: {self.num_points}")
        print(f"Total(s):\nRasterization: {times[0]:.3f}, Backward: {times[1]:.3f}")
        print(
            f"Per step(s):\nRasterization: {times[0] / iterations:.5f}, Backward: {times[1] / iterations:.5f}"
        )

        out_img = torch.clamp(out_img, min=0.0, max=1.0)
        out_img = (out_img.detach().cpu().numpy() * 255.0).astype(np.uint8)
        #plt.imshow(out_img)
        #plt.show()

        means = self.means
        rotations = torch.sigmoid(self.rotations) * torch.pi
        scales = torch.relu(self.scales)+1.
        colors = torch.sigmoid(self.rgbs)

        return ll1_loss, shape_loss,out_img, means, scales, rotations, colors


def image_path_to_tensor(image_path: str, height, width):
    import torchvision.transforms as transforms

    img = Image.open(image_path)

    # width, height = img.size
    if width is not None and height is not None:
        # transform = transforms.ToTensor()
        transform = transforms.Compose([
            transforms.Resize([width, height]),
            transforms.ToTensor()
        ])
    else:
        transform = transforms.ToTensor()
    img_tensor = transform(img).permute(1, 2, 0)[..., :3]
    return img_tensor, width, height


def main(
        height: int = None,
        width: int = None,
        num_points: int = 4000,
        save_imgs: bool = False,
        img_path: str = '/home/ubuntu/datasets/Counting/UCF-Train-Val-Test/val/IMG_0007.jpg',
        iterations: int = 4000,
        lr: float = 0.01,
        gt_points: bool = True,
        bk_mask: bool = True,
        bk_num: int = 0,
) -> None:
    if img_path:
        gt_image, width, height = image_path_to_tensor(img_path, height, width)
        if gt_points:
            point_path = img_path.replace('.jpg', '.npy')
            points = torch.tensor(np.load(point_path))
        else:
            points = None
    else:
        gt_image = torch.ones((height, width, 3)) * 1.0
        # make top left and bottom right red, blue
        gt_image[: height // 2, : width // 2, :] = torch.tensor([1.0, 0.0, 0.0])
        gt_image[height // 2:, width // 2:, :] = torch.tensor([0.0, 0.0, 1.0])
        points = None
    print(points)
    trainer = SimpleTrainer(gt_image=gt_image, num_points=num_points, gt_points=points, bk_num=bk_num, use_bk_mask=bk_mask)
    trainer.train(
        iterations=iterations,
        lr=lr,
        save_imgs=save_imgs
    )


if __name__ == "__main__":
    tyro.cli(main)

