# GST: 2D Gaussians Spatial Transport for Point-supervised Density-Regression
Official Implement of AAAI 2026 paper '**2D Gaussians Spatial Transport for Point-supervised Density Regression**' (GST).

[![arXiv](https://img.shields.io/badge/GST-2511.14477-b31b1b)](https://arxiv.org/abs/2511.14477)

<div align="justify">
This paper introduces Gaussian Spatial Transport (GST), a novel framework that leverages Gaussian splatting to facilitate transport from the probability measure in the image coordinate space to the annotation map. We propose a Gaussian splatting-based method to estimate pixel-annotation correspondence, which is then used to compute a transport plan derived from Bayesian probability. To integrate the resulting transport plan into standard network optimization in typical computer vision tasks, we derive a loss function that measures discrepancy after transport. Extensive experiments on representative computer vision tasks, including crowd counting and landmark detection, validate the effectiveness of our approach. Compared to conventional optimal transport schemes, GST eliminates iterative transport plan computation during training, significantly improving efficiency.
</div>


## Overview

<div align="justify">
We propose Gaussian Spatial Transport (GST), a novel framework that efficiently transports probability measures from image coordinates to annotation space. 
At its core, GST establishes an interpretable probabilistic correspondence between pixels and annotations by leveraging 2D Gaussian Splatting. 
This correspondence is then distilled into a fixed transport kernel, a matrix that encodes the static mapping from each pixel to all target annotations, which can be pre-computed before training. 
During optimization, the model’s predicted density map is pushed forward to the space of ground truth annotations via a single matrix multiplication with this pre-computed kernel. 
Whereas OT-based methods require iteratively solving a costly optimization problem to determine the transport plan and compute the loss, our loss is simply the discrepancy between the transported density and the ground truth. 
Extensive experiments demonstrate that GST achieves a strong balance of high accuracy and computational efficiency.
</div>
<br>
<br>
<div>
  <img align="center" src=".\figs\related_works.png" alt="kodak_fitting" />

  <sub align="left">**Comparison of Direct Regression, OT, and the proposed GST.** Direct Regression (a) rely on handcrafted Gaussian kernels to generate pseudo-ground-truth density maps for direct pixel-to-pixel regression, creating a critical dependency on the quality of these synthetic approximations. 
  In contrast, Optimal Transport (b) is dynamic and more sophisticated; it iteratively re-calculates the best matching plan and its associated cost within every single training loop, making it accurate but computationally slow. 
  Our proposed GST (c) offers a "best-of-both-worlds" solution: it pre-computes a high-quality, fixed transport pattern based on Gaussian Splatting once before training begins. This allows the loss to be calculated in a single, highly efficient step during training, achieving the accuracy of a transport-based method without the costly iterative overhead.</sub>
</div>
<br>

<div>
  <img align="center" src=".\figs\pipeline.png" alt="kodak_fitting" />

  <sub align="left">**The GST Pipeline comprises two main components: transport kernel generation and model training.** First, the transport kernel K is generated before training by reconstructing the RGB image via 2D Gaussian splatting and establishing pixel-toannotation correspondences (Eq. 7) to then form K (Eq. 5). 
Second, during training, K transports the estimated density map to annotations, allowing for the computation of the transported mass discrepancy loss (Eq. 6).</sub>
</div>
<br>

<div>
  <img align="center" src=".\figs\count_vis_suppl.png" alt="kodak_fitting" />

  <sub align="left">**Visualizations.** Our GST yielded a more accurate count estimate and a sharp density distribution as OT.
</div>


## Requirements
- python 3.9 (We recommend to use Anaconda, since many python libs like numpy and sklearn are needed in our code.)

- [PyTorch 2.0.1-cuda11.8](https://pytorch.org/) (we run the code under version 2.0.1 with CUDA-11.8)  

- [gsplat](https://github.com/infinite0522/2d_image_gsplat): we modified [original gsplat](https://github.com/nerfstudio-project/gsplat) for Gaussian Splatting on 2D images. Please refer to [here](https://github.com/infinite0522/2d_image_gsplat) to install the package. 


## Dataset preparation

1. Dataset download

+ QNRF can be downloaded [here](https://www.crcv.ucf.edu/data/ucf-qnrf/)

+ JHU-Crowd++ can be downloaded [here](http://www.crowd-counting.com/)

+ NWPU can be downloaded [here](https://www.crowdbenchmark.com/nwpucrowd.html)


2. Data preprocess

Due to large sizes of images in the datasets, we preprocess these datasets.

```
# for qnrf dataset
python preprocess_dataset_ucf.py --input-dataset-path <original data directory> --output-dataset-path <processed data directory> 
```

The folder of the datasets is as follows (taking qnrf for example):
```
UCF-Train-Val-Test
├── train
│   ├── IMG_0001.jpg
│   ├── IMG_0001.npy
│   ├── IMG_0002.jpg
│   ├── IMG_0002.jpg
│   └── ...
├── val
│   ├── IMG_0007.jpg
│   ├── IMG_0007.npy
│   ├── IMG_0012.jpg
│   ├── IMG_0012.jpg
│   └── ...
├── test
... ...
```

## GST Pre-computation
1. Perform 2d Gaussian Splatting for all images in the datasets, to obtain and store Gaussian primitives.
```
# for qnrf dataset
python 2dgs_counting_1gs_gsplat.py --config 'GS2D/config_qnrf_gsplat.yml'
```
The hyperparameters for other datasets are identical, just modify the dataset path in ``config_qnrf_gsplat.yml''.

2. Process distorted Gaussians
```
python GS2D/process_gaussians.py
```
After GST Pre-computation, the folder of the datasets is as follows:
```
UCF-Train-Val-Test
├── train
│   ├──gs_params
│   │  ├──IMG_0001_gs_params.h5
│   │  ├──IMG_0002_gs_params.h5
│   │  └──...
│   ├── IMG_0001.jpg
│   ├── IMG_0001.npy
│   ├── IMG_0002.jpg
│   ├── IMG_0002.jpg
│   └── ...
├── val
│   ├── IMG_0007.jpg
│   ├── IMG_0007.npy
│   ├── IMG_0012.jpg
│   ├── IMG_0012.jpg
│   └── ...
├── test
... ...
```
We have released our processed Gaussian primitives [here](https://1drv.ms/f/c/a20a9dcd7478ce96/IgCr1ywU6f1OTpqOO9ibN-IgAaO9FYyMO1ZaMF2aTviVqLo?e=Rn3eC0), which can be used directly for task-tailored network training.

## Task-tailored Network Training

```
python train_gs.py --model-name <backbon: vgg or vgg_trans> --dataset <dataset name: qnrf, jhu or nwpu> --data-dir <path to dataset> --device <gpu device id>
```

## Evaluation

download our pretrained checkpoints [here](https://1drv.ms/f/c/a20a9dcd7478ce96/IgDd_V-xWXRXTonFULYlVItNASwFqqr3UoRm_GQGi20KrXc?e=qVg4VL) and prepare the dataset.
```
python test.py --save-path <path of the model to be evaluated> --data-dir <directory for the dataset> --device <gpu device id>
```

## License

Please check the MIT  [license](./LICENSE) that is listed in this repository.

## Acknowledgments

We thank the following repos providing helpful components/functions in our work.

- [gsplat](https://github.com/infinite0522/2d_image_gsplat)  
- [Bayesian-Crowd-Counting](https://github.com/zhiheng-ma/Bayesian-Crowd-Counting)

## Citation

If you use any content of this repo for your work, please cite the following bib entry:
```
@inproceedings{shang2026GST,
  title={2D Gaussians Spatial Transport for Point-supervised Density-Regression},
  author={Shang, Miao and Hong, Xiaopeng},
  booktitle={Proceedings of the 40th Annual AAAI Conference on Artificial Intelligence},
  year={2026}
}
```

