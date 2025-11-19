import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import softmax
from torch.nn import Module



def reset_func(x):
    return torch.where((x >= 0) & (x < 5), torch.tensor(2.),
                       torch.where((x >= 5) & (x < 10), torch.tensor(7.),
                            torch.where((x >= 10) & (x < 15), torch.tensor(12.),
                                torch.where((x >= 15) & (x < 20), torch.tensor(17.),
                                    torch.where((x >= 20) & (x < 25), torch.tensor(22.),
                                        torch.where((x >= 25) & (x < 30), torch.tensor(27.),
                                            torch.where(x >= 30, torch.tensor(32.),x)))))))

def reset_func_vec(x):

    return torch.where((x >= 0) & (x < 5), torch.tensor(2.),
                       torch.where((x >= 5) & (x < 10), torch.tensor(7.),
                            torch.where((x >= 10) & (x < 15), torch.tensor(12.),
                                torch.where((x >= 15) & (x < 20), torch.tensor(17.),
                                    torch.where((x >= 20) & (x < 25), torch.tensor(22.),
                                        torch.where((x >= 25) & (x < 30), torch.tensor(27.),
                                            torch.where(x >= 30, torch.tensor(32.),x)))))))

class Post_Prob_GS(Module):
    def   __init__(self, c_size, stride, background_ratio, use_background, post_min, post_max, scale_ratio, device, cut_off=None, scale_standard=False):
        super(Post_Prob_GS, self).__init__()
        assert c_size[0] % stride == 0 & c_size[1] % stride == 0

        self.bg_ratio = background_ratio
        self.device = device

        self.d_ratio = stride
        self.post_min = post_min
        self.post_max = post_max
        self.scale_ratio = scale_ratio
        self.cut_off = cut_off

        h, w = c_size[:2]
        xx = torch.arange(0, w, step=stride, dtype=torch.float32, device=device) + stride / 2  # w
        yy = torch.arange(0, h, step=stride, dtype=torch.float32, device=device) + stride / 2  # h
        x, y = torch.meshgrid(xx, yy, indexing="xy")  # (h,w)
        self.x = x.to(self.device).unsqueeze(0)
        self.y = y.to(self.device).unsqueeze(0)

        self.softmax = torch.nn.Softmax(dim=0)
        self.use_bg = use_background
        self.scale_standard = scale_standard


    def forward(self, scale, rotation, points, st_sizes):
        num_points_per_image = [len(points_per_image) for points_per_image in points]
        all_points = torch.cat(points, dim=0)
        all_scales = torch.cat(scale, dim=0)
        all_rotations = torch.cat(rotation, dim=0)
        bt_size = len(all_points)
        if len(all_points) > 0:
            all_scales = all_scales.view(bt_size, 2)
            all_rotations = all_rotations.view(bt_size, 1)
            all_scales = torch.clamp(self.scale_ratio * all_scales, min=self.d_ratio / self.post_min, max=self.post_max)

            scale_matrices = torch.diag_embed(all_scales)
            cosines = torch.cos(all_rotations)
            sines = torch.sin(all_rotations)
            rot_matrices = torch.cat([cosines, -sines, sines, cosines], 1).reshape(-1, 2, 2)
            covariances = (rot_matrices @ scale_matrices @ torch.transpose(scale_matrices, -2, -1) @ torch.transpose(
                rot_matrices, -2, -1).to(self.device))
            covariances = covariances.unsqueeze(1).unsqueeze(1)

            inv_covariances = torch.inverse(covariances)

            xy = torch.stack([self.x, self.y], dim=-1).repeat(bt_size,1,1,1)  # torch.Size([1, w / stride, h / stride, 2])
            xy = (xy - all_points.unsqueeze(1).unsqueeze(1))
            dis = torch.einsum('...i,...i->...', xy, xy).view(bt_size,-1)
            #print(xy.shape)
            z = torch.einsum('b...i,b...ij,b...j->b...', xy, -0.5 * inv_covariances,
                             xy).view(bt_size,-1)

            z_list = torch.split(z, num_points_per_image)
            covariances_list = torch.split(covariances, num_points_per_image)
            dis_list = torch.split(dis, num_points_per_image)
            prob_list = []
            for dis, z, covariances, st_size in zip(dis_list, z_list, covariances_list, st_sizes):
                if len(z) > 0:

                    det_covariances = torch.det(covariances).view(-1, 1)
                    likelihood = (z - 0.5 * torch.log(det_covariances))

                    if self.use_bg and self.cut_off is not None:
                        min_z, index = torch.max(z, dim=0, keepdim=True)
                        d = -1.0 * self.cut_off * self.cut_off / 2.0
                        index = index.squeeze(0)
                        bk_z = (d - min_z).squeeze(0)
                        bk_likelihood = (bk_z - 1.0 * torch.log(det_covariances.squeeze(1)[index] / self.bg_ratio)).unsqueeze(0)
                        likelihood = torch.cat([likelihood, bk_likelihood], 0)

                    post_prob = self.softmax(likelihood)

                else:
                    post_prob = None

                prob_list.append(post_prob)
        else:
            prob_list = []
            for _ in range(len(points)):
                prob_list.append(None)
        return prob_list
