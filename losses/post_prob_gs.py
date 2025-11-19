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
        # coordinate is same to image space, set to constant since crop size is same
        #self.cood = torch.arange(0, c_size, step=stride,
        #                         dtype=torch.float32, device=device) + stride / 2

        self.d_ratio = stride
        self.post_min = post_min
        self.post_max = post_max    # 其他数据集要修改：每张图片st_size不同，post_max不同
        self.scale_ratio = scale_ratio
        self.cut_off = cut_off

        h, w = c_size[:2]
        xx = torch.arange(0, w, step=stride, dtype=torch.float32, device=device) + stride / 2  # w
        yy = torch.arange(0, h, step=stride, dtype=torch.float32, device=device) + stride / 2  # h
        x, y = torch.meshgrid(xx, yy, indexing="xy")  # (h,w)
        self.x = x.to(self.device).unsqueeze(0)
        self.y = y.to(self.device).unsqueeze(0)
        #print(x, y, x.shape, y.shape)
        """
        print(self.cood)
        tensor([4., 12., 20., 28., 36., 44., 52., 60., 68., 76., 84., 92.,
                100., 108., 116., 124., 132., 140., 148., 156., 164., 172., 180., 188.,
                196., 204., 212., 220., 228., 236., 244., 252., 260., 268., 276., 284.,
                292., 300., 308., 316., 324., 332., 340., 348., 356., 364., 372., 380.,
                388., 396., 404., 412., 420., 428., 436., 444., 452., 460., 468., 476.,
                484., 492., 500., 508.], device='cuda:0')
        """

        #self.cood.unsqueeze_(0)
        self.softmax = torch.nn.Softmax(dim=0)
        self.use_bg = use_background
        self.scale_standard = scale_standard


    def forward(self, scale, rotation, points, st_sizes):
        num_points_per_image = [len(points_per_image) for points_per_image in points]
        all_points = torch.cat(points, dim=0)
        all_scales = torch.cat(scale, dim=0)
        all_rotations = torch.cat(rotation, dim=0)
        bt_size = len(all_points)
        #print(all_rotations.shape, all_scales.shape)
        #print(all_scales,all_rotations)

        if len(all_points) > 0:
            all_scales = all_scales.view(bt_size, 2)
            all_rotations = all_rotations.view(bt_size, 1)
            all_scales = torch.clamp(self.scale_ratio * all_scales, min=self.d_ratio / self.post_min, max=self.post_max)  #4.24) #  4.24=6/sqrt(2)

            #if self.scale_standard:
            #    all_scales = reset_func(all_scales)

            #print(all_scales[:10])
            scale_matrices = torch.diag_embed(all_scales)
            #cosines = torch.cos(all_rotations * torch.pi)      # shanghaiB开始在cpu版本高斯泼溅学习存储的参数需要*pi
            #sines = torch.sin(all_rotations * torch.pi)      # shanghaiB需要*pi
            cosines = torch.cos(all_rotations)
            sines = torch.sin(all_rotations)
            rot_matrices = torch.cat([cosines, -sines, sines, cosines], 1).reshape(-1, 2, 2)
            covariances = (rot_matrices @ scale_matrices @ torch.transpose(scale_matrices, -2, -1) @ torch.transpose(
                rot_matrices, -2, -1).to(self.device))
            covariances = covariances.unsqueeze(1).unsqueeze(1)
            #print(covariances.shape)

            inv_covariances = torch.inverse(covariances) # 逆矩阵

            xy = torch.stack([self.x, self.y], dim=-1).repeat(bt_size,1,1,1)  # torch.Size([1, w / stride, h / stride, 2])
            xy = (xy - all_points.unsqueeze(1).unsqueeze(1))
            dis = torch.einsum('...i,...i->...', xy, xy).view(bt_size,-1)
            #print(xy.shape)
            z = torch.einsum('b...i,b...ij,b...j->b...', xy, -0.5 * inv_covariances,
                             xy).view(bt_size,-1)  # 高斯核每个坐标点的指数部分 [B, w / stride, h / stride]

            #likelihood = (torch.exp(z) / (
            #        torch.sqrt(torch.det(covariances)).view(bt_size, 1,1)
            #        )) #.view(bt_size, -1)

            # likelihood = (z - 0.5 * torch.log(torch.det(covariances)).view(bt_size, 1,1)).view(bt_size,-1)
            # print(likelihood[0].view(96,128))

            z_list = torch.split(z, num_points_per_image)
            covariances_list = torch.split(covariances, num_points_per_image)
            dis_list = torch.split(dis, num_points_per_image)
            prob_list = []
            # cost_list = []
            for dis, z, covariances, st_size in zip(dis_list, z_list, covariances_list, st_sizes):
                if len(z) > 0:

                    det_covariances = torch.det(covariances).view(-1, 1)
                    likelihood = (z - 0.5 * torch.log(det_covariances))
                    #print("88888888888888888888888888888888888")
                    #print(likelihood.shape)

                    """ bayes background
                    if self.use_bg and self.cut_off is None:
                        min_dis = torch.clamp(torch.min(-z * (2.0 * 8.0 ** 2), dim=0, keepdim=True)[0], min=0.0)
                        d = st_size * self.bg_ratio
                        bg_dis = (d - torch.sqrt(min_dis)) ** 2
                        likelihood = torch.cat([likelihood, - bg_dis / (2.0 * 8.0 ** 2) - 0.5 * np.log(64.0)],
                                               0)  # concatenate background distance to the last
                        '''

                        min_dis = torch.clamp(torch.min(dis, dim=0, keepdim=True)[0], min=0.0)
                        print(dis)
                        print(min_dis)
                        d = st_size * self.bg_ratio
                        bg_dis = (d - torch.sqrt(min_dis)) ** 2
                        dis = torch.cat([dis, bg_dis], 0)  # concatenate background distance to the last
                        likelihood = -dis / (2.0 * 8.0 ** 2)
                        '''
                    """


                    if self.use_bg and self.cut_off is not None:
                        #print(z)
                        """ version 1"""
                        min_z, index = torch.max(z, dim=0, keepdim=True)
                        d = -1.0 * self.cut_off * self.cut_off / 2.0
                        index = index.squeeze(0)
                        bk_z = (d - min_z).squeeze(0)
                        bk_likelihood = (bk_z - 1.0 * torch.log(det_covariances.squeeze(1)[index] / self.bg_ratio)).unsqueeze(0)
                        likelihood = torch.cat([likelihood, bk_likelihood], 0)

                    """ bk visualization
                    print(likelihood.shape)
                    a = likelihood[-1].view(64,64).cpu().detach().numpy()
                    print("aaaaaaaaaaaaaaaaaaaaaaaaaaa", a.max())
                    normalized_a = (a - np.min(a)) / (np.max(a) - np.min(a))
                    plt.imshow(normalized_a)
                    plt.show()
                    """

                    ''' temp: for pseudo gaussian generation

                    z = torch.where(z > - 9.0 / 2.0, z * torch.ones_like(z), torch.zeros_like(z) - 1e+5)
                    bk_z = torch.zeros((1, z.shape[1]), device=self.device).float() - 9.0 /2.0
                    z = torch.cat([z, bk_z], dim=0)
                    post_prob = torch.exp(z)
                    '''


                    post_prob = self.softmax(likelihood)

                    """ entropy visualization
                    prob = post_prob.view(-1, 64, 64) + 1e-10
                    entropy = - prob * torch.log(prob)
                    entropy_img = entropy.sum(dim=0)
                    entropy_img = entropy_img.cpu().detach().numpy()
                    entropy_img = np.array(np.squeeze(plt.cm.jet(entropy_img)[:, :, :3]))
                    plt.imshow(entropy_img)
                    plt.show()
                    """


                    # print(post_prob.shape)
                else:
                    post_prob = None
                    #cost = 0.

                prob_list.append(post_prob)
                #cost_list.append(cost)
        else:
            prob_list = []
            #cost_list = []
            for _ in range(len(points)):
                prob_list.append(None)
                #cost_list.append(0.)
        #print("88888888888888888888888888888888888")
        #print(prob_list[0].shape)
        #print(len(prob_list))
        return prob_list #,cost_list
