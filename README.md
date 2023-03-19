# EMO

Official [PyTorch](https://pytorch.org/) implementation of "[Rethinking Mobile Block for Efficient Neural Models](https://arxiv.org/abs/2301.01146)".<br>
[Jiangning Zhang](https://zhangzjn.github.io/), [Xiangtai Li](https://lxtgh.github.io/), Jian Li, Liang Liu, Zhucun Xue, Boshen Zhang, Zhengkai Jiang, Tianxin Huang, [Yabiao Wang](https://scholar.google.com.hk/citations?user=xiK4nFUAAAAJ&hl=zh-CN&oi=ao), and [Chengjie Wang](https://scholar.google.com.hk/citations?user=fqte5H4AAAAJ&hl=zh-CN&oi=ao)

> **Abstract** This paper focuses on developing modern, efficient, lightweight models for dense predictions while trading off parameters, FLOPs, and performance. Inverted Residual Block (IRB) serves as the infrastructure for lightweight CNNs, but no counterpart has been recognized by attention-based studies. This work rethinks lightweight infrastructure from efficient IRB and effective components of Transformer from a unified perspective, extending CNN-based IRB to attention-based models and abstracting a one-residual Meta Mobile Block (MMB) for lightweight model design. Following simple but effective design criterion, we deduce a modern **I**nverted **R**esidual **M**obile **B**lock (iRMB) and build a ResNet-like Efficient MOdel (EMO) with only iRMB for down-stream tasks. Extensive experiments on ImageNet-1K, COCO2017, and ADE20K benchmarks demonstrate the superiority of our EMO over state-of-the-art methods, *e.g.*, EMO-1M/2M/5M achieve 71.5, 75.1, and 78.4 Top-1 that surpass equal-order CNN-/Attention-based models, while trading-off the parameter, efficiency, and accuracy well: running 2.8-4.0 $\times \uparrow$ faster than EdgeNeXt on iPhone14. Codes and models are available in the supplementary material.

<div align="center">
  <img src="resources/meta_mobile_block.png" width="800px" />
</div>

> **Left**: Abstracted unified ***Meta-Mobile Block*** from *Multi-Head Self-Attention* and *Feed-Forward Network* in Transformer as well as efficient *Inverted Residual Block* in MobileNet-v2. Absorbing the experience of light-weight CNN and Transformer, an efficient but effective ***EMO*** is designed based on deduced iRMB.<br>
> **Right**: *Performance* vs. *FLOPs* comparisons with SoTA Transformer-based methods.

>**This paper is still under a review process, and we will release the code according to the review results. Please stay tuned.**

