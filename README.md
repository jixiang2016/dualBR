## Motion Blur Decomposition with Cross-shutter Guidance
![visitors](https://visitor-badge.laobi.icu/badge?page_id=jixiang2016/dualBR)  [Arxiv](https://drive.google.com/file/d/1l0GMiv2xMcVaSuIY4E7f3zPljtRq1mju/view) | [Paper]() | [Supp]( )

Xiang Ji, Haiyang Jiang, Yinqiang Zheng

The University of Tokyo


This repository provides the official PyTorch implementation of the paper.

#### TL;DR
This paper explores a novel in-between exposure mode called global reset release (GRR) shutter, which produces GS-like blur but with row-dependent blur magnitude. We take advantage of this unique characteristic of GRR to explore the latent frames within a single image and restore a clear counterpart by relying only on these latent contexts.

<img width="700" alt="image" src="docs/shutter_modes.png">


## Dependencies
1. Python and Pytorch
- Pyhotn=3.8 (Anaconda recommended)
- Pytorch=1.11.0
- CUDA=11.3/11.4
``` shell
conda create -n dualbr python=3.8
conda activate dualbr
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
```
2. Other packages
``` shell
pip install -r requirements.txt
```
