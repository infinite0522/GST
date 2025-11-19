from utils.regression_trainer_cosine_gs import RegTrainer
from utils.helper import setup_seed
import argparse
import os
import torch
args = None

def parse_args():
    parser = argparse.ArgumentParser(description='Train ')
    parser.add_argument('--model-name', default='vgg19_trans', help='the name of the model')
    parser.add_argument('--data-dir', default=r'/home/home/shangmiao/datasets/Counting/NWPU-Train-Val-Test',
                        help='training data directory')
    parser.add_argument('--save-dir', default='./saved_models/MAN_gs/nwpu',
                        help='directory to save models.')
    parser.add_argument('--dataset', type=str, default='nwpu',
                        help='the dataset')

    parser.add_argument('--save-all', type=bool, default=False,
                        help='whether to save all best model')

    parser.add_argument('--lr', type=float, default=5*1e-6,
                        help='the initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        help='the weight decay')
    parser.add_argument('--resume', default='',
                        help='the path of resume training model')
    parser.add_argument('--max-model-num', type=int, default=1,
                        help='max models num to save ')
    parser.add_argument('--max-epoch', type=int, default=1200,
                        help='max training epoch')
    parser.add_argument('--val-epoch', type=int, default=5,
                        help='the num of steps to log training information')
    parser.add_argument('--val-start', type=int, default=600,
                        help='the epoch start to val')
    parser.add_argument('--topk', type=float, default=0.3,
                        help='top k')
    parser.add_argument('--usenum', type=int, default=18,
                        help='usenum')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='train batch size')
    parser.add_argument('--device', default='2', help='assign device')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='the num of training process')

    parser.add_argument('--is-gray', type=bool, default=False,
                        help='whether the input image is gray')
    parser.add_argument('--crop-size', type=int, default=512,
                        help='the crop size of the train image')
    parser.add_argument('--downsample-ratio', type=int, default=8,
                        help='downsample ratio')

    parser.add_argument('--use-background', type=bool, default=True,
                        help='whether to use background modelling')
    parser.add_argument('--sigma', type=float, default=8.0,
                        help='sigma for likelihood')
    parser.add_argument('--background-ratio', type=float, default=0.015,
                        help='background ratio')

    parser.add_argument('--post-min', type=float, default=1.0,
                        help='post min')
    parser.add_argument('--post-max', type=float, default=20.0,
                        help='post max')
    parser.add_argument('--scale-ratio', type=float, default=1.0,
                        help='scale ratio')
    parser.add_argument('--cut-off', type=float, default=3.0,
                        help='cut off')

    parser.add_argument('--seed', type=int, default=4073,
                        help='random seed')
    parser.add_argument('--gs-path', type=str, default='gs_params_2',
                        help='path of gs_params files')
    args = parser.parse_args()

    if args.dataset.lower() == 'qnrf':
        args.crop_size = 512
    elif args.dataset.lower() == 'nwpu':
        args.crop_size = 384
        args.val_epoch = 50
    elif args.dataset.lower() == 'sha':
        args.crop_size = 256
    elif args.dataset.lower() == 'shb':
        args.crop_size = 512
    elif args.dataset.lower() == 'jhu':
        args.crop_size = 512
    else:
        raise NotImplementedError

    return args


if __name__ == '__main__':
    args = parse_args()
    setup_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # set vis gpu
    trainer = RegTrainer(args)
    trainer.setup()
    trainer.train()
