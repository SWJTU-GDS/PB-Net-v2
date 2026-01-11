# Deep Physiological-Behavioral Representation Learning for Video-Based Hand Gesture Authentication
Pytorch Implementation of paper:

> **Deep Physiological-Behavioral Representation Learning for Video-Based Hand Gesture Authentication**
>
> Wenwei Song, Xiaorong Gao, Yufeng Zhang, Jinlong Li, Wenxiong Kang, and Zhixue Wang\*.

## Main Contribution
Dynamic hand gestures encode rich physiological and behavioral characteristics, providing a promising biometric trait for reliable authentication. Existing studies primarily improve video-based gesture authentication by designing network architectures, constructing behavioral pseudo-modalities, and optimizing loss functions. Following this paradigm, PB-Net adopts a decoupled analysis and complementary fusion strategy for the two characteristics, achieving competitive performance. However, its modeling of fine-grained identity characteristics remains limited. In this work, we revisit PB-Net and propose PB-Net v2 by rethinking the modeling requirements of physiological and behavioral characteristics. Specifically, we refine the data-tailoring strategy, including behavioral pseudo-modality design, to reduce redundancy while preserving richer identity information. We then enhance the physiological and behavioral branches to extract more complementary spatiotemporal physiological features and more stable behavioral representations, respectively. Moreover, we improve the feature fusion module to mitigate branch-specific bias while facilitating reliability-aware feature fusion. Extensive experiments on the SCUT-DHGA dataset demonstrate the effectiveness of the proposed improvements. PB-Net v2 consistently achieves the lowest equal error rates among 21 state-of-the-art models under four evaluation protocols.
<div align="center">
 <p align="center">
  <img src="https://raw.githubusercontent.com/SWJTU-GDS/PB-Net-v2/master/img/PBNet-v2.png" />
 </p>
</div>

Overall architecture of PB-Net v2. C1 denotes the Conv1 layer of ResNet, and LxBy indicates the y-th Block in the x-th ResNet Layer. TC represents Temporal Convolution, while TM denotes Temporal Max Pooling. Norm indicates L2 normalization. $\mathcal{L}_1$-$\mathcal{L}_3$ correspond to three AMSoftmax loss functions.

## Comparisons with SOTAs
To comprehensively evaluate the effectiveness of PB-Net v2, we compare it with 21 SOTA video understanding models on the SCUT-DHGA dataset. The performance of some representative models are shown in the following figure. The EERs shown in the figure are all average values over six test configurations on the cross session.

 <div align="center">
 <p align="center">
  <img src="https://raw.githubusercontent.com/SCUT-BIP-Lab/PB-Net-v2/master/img/sota_comparison.png" />
 </p>
</div>

Comparison of PB-Net v2 with representative models under MG and UMG protocols using the AMSoftmax loss. The values at the bottom and top of each bar indicate the MG and UMG EERs, respectively.

## Dependencies
Please make sure the following libraries are installed successfully:
- [PyTorch](https://pytorch.org/) >= 2.2.2

## How to use
This repository is a demo of PB-Net-v2. Through debugging ([main.py](/main.py)), you can quickly understand the 
configuration and building method of [PB-Net-v2](/model/PBNet.py).

If you want to explore the entire dynamic hand gesture authentication framework, please refer to our pervious work [SCUT-DHGA](https://github.com/SCUT-BIP-Lab/SCUT-DHGA).
