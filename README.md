# AdaCS: Adaptive Compressive Sensing With Restricted Isometry Property-Based Error-Clamping (TPAMI 2024) 

This repository is the pytorch implement of [AdaCS: Adaptive Compressive Sensing With Restricted Isometry Property-Based Error-Clamping](https://ieeexplore.ieee.org/document/10412658)

Chenxi Qiu and [Xuemei Hu](https://scholar.google.com.hk/citations?hl=zh-CN&user=yZauWzEAAAAJ)

*School of Electrical Science and Engineering, Nanjing University, Jiangsu, China.*

## Abstract

Scene-dependent adaptive compressive sensing (CS) has been a long pursuing goal that has huge potential to significantly improve the performance of CS. However, with no access to the ground truth, how to design the scene-dependent adaptive strategy is still an open problem. In this paper, a restricted isometry property (RIP) condition-based error-clamping is proposed, which could directly predict the reconstruction error, i.e. the difference between the current-stage reconstructed image and the ground truth image, and adaptively allocate more samples to regions with larger reconstruction error at the next sampling stage. Furthermore, we propose a CS reconstruction network composed of Progressively inverse transform and Alternating Bi-directional Multi-grid Network, named PiABM-Net, that could efficiently utilize the multi-scale information for reconstructing the target image. The effectiveness of the proposed adaptive and cascaded CS method is demonstrated with extensive quantitative and qualitative experiments, compared with the state-of-the-art CS algorithms.

## Overview

### Cascaded AdaCS framework
![Cascaded AdaCS framework](figs/AdaCS.png)
 
### PiABM-Net
![PiABM-Net](figs/PiABM-Net.png)

## Environmental Requirements
- Python == 3.8.16
- Pytorch == 2.0.1

## Test   

Download the [pretrained weights](https://box.nju.edu.cn/d/89dca6f22250415c9768/) and put it into `./weights/`, then run:

## Command
### Train
`python train.py --gpu_list 0`
### Test
`python test.py --test_stage 0/1/2/3/4 --test_name Set11`

## Citation

If you find the code helpful in your research or work, please cite the following paper:

```
@article{qiu2024adacs,
  title={AdaCS: Adaptive Compressive Sensing with Restricted Isometry Property-Based Error-clamping},
  author={Qiu, Chenxi and Hu, Xuemei},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2024},
  publisher={IEEE}
}
```
