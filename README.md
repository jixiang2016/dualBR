## Motion Blur Decomposition with Cross-shutter Guidance
![visitors](https://visitor-badge.laobi.icu/badge?page_id=jixiang2016/dualBR)  [Arxiv](http://arxiv.org/abs/2404.01120) | [Paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Ji_Motion_Blur_Decomposition_with_Cross-shutter_Guidance_CVPR_2024_paper.pdf) | [Supp](https://openaccess.thecvf.com/content/CVPR2024/supplemental/Ji_Motion_Blur_Decomposition_CVPR_2024_supplemental.pdf) | [Project page](https://jixiang2016.github.io/dualBR_site)

Xiang Ji, Haiyang Jiang, Yinqiang Zheng

The University of Tokyo


This repository provides the official PyTorch implementation of the paper.

#### TL;DR
Inspired by the complementary exposure characteristics of a global shutter (GS) camera and a rolling shutter (RS) camera, we propose a dual Blur-RS setting to solve the motion ambiguity of blur decomposition. As shown in the Figure below, the RS view not only provides local details but also implicitly captures the temporal order of latent frames. Meanwhile, GS view could be exploited to mitigate the initial-state ambiguity from RS counterpart.

<br/>
<img width="800" alt="image" src="docs/img.png">

 
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

## Data and Pretrained Model
- Download datasets [realBR](https://drive.google.com/file/d/1s9t6DHHfzB-o6pDhg4vqcgcFR6gdLxqj/view?usp=sharing) and synthetic data [GOPRO-VFI_copy](https://drive.google.com/file/d/1AaI1DD0SYECfL8GXFuCR6a4tZieRCMMe/view?usp=sharing) based on [GOPRO](https://drive.google.com/file/d/1rJTmM9_mLCNzBUUhYIGldBYgup279E_f/view?usp=sharing).  <!--   coming soon   -->
- Unzip them under a specified directory by yourself.
- Please download checkpoints from this [link](https://drive.google.com/drive/folders/1VDj-TLQxNCm3eIgCt5C-dJbYPDNzVLDO?usp=sharing) and put them under root directory of this project.

## Test
To test model, please run the command below:
``` shell
bash ./test.sh       ### Please specify your data directory, output path in the script
```
## Train
To train model, please run the command below:
``` shell
bash ./train.sh       ### Please refer to the script for more info.
```

## Acknowledgement
This project is implemented by partially referring to the code of work below:
- [RIFE](https://github.com/megvii-research/ECCV2022-RIFE)
