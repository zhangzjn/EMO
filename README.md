# EMO

--- 
Official [PyTorch](https://pytorch.org/) implementation of "[Rethinking Mobile Block for Efficient Neural Models](https://arxiv.org/abs/2301.01146)".

> **Abstract** This paper focuses on designing efficient models with low parameters and FLOPs for dense predictions. Even though CNN-based lightweight methods have achieved stunning results after years of research, trading-off model accuracy and constrained resources still need further improvements. This work rethinks the essential unity of efficient Inverted Residual Block in MobileNetv2 and effective Transformer in ViT, inductively abstracting a general concept of Meta-Mobile Block, and we argue that the specific instantiation is very important to model performance though sharing the same framework. Motivated by this phenomenon, we deduce a simple yet efficient modern **I**nverted **R**esidual **M**obile **B**lock (iRMB) for mobile applications, which absorbs CNN-like efficiency to model short-distance dependency and Transformer-like dynamic modeling capability to learn long-distance interactions. Furthermore, we design a ResNet-like 4-phase **E**fficient **MO**del (EMO) based only on a series of iRMBs for dense applications. Massive experiments on ImageNet-1K, COCO2017, and ADE20K benchmarks demonstrate the superiority of our EMO over state-of-the-art methods, e.g., our EMO-1M/2M/5M achieve 71.5, 75.1, and 78.4 Top-1 that surpass **SoTA** CNN-/Transformer-based models, while trading-off the model accuracy and efficiency well. Codes and models are available in the supplementary material.

<div align="center">
  <img src="resources/meta_mobile_block.png" width="800px" />
</div>

> **Left**: Abstracted unified ***Meta-Mobile Block*** from *Multi-Head Self-Attention* and *Feed-Forward Network* in Transformer as well as efficient *Inverted Residual Block* in MobileNet-v2. Absorbing the experience of light-weight CNN and Transformer, an efficient but effective ***EMO*** is designed based on deduced iRMB.<br>
> **Right**: *Performance* vs. *FLOPs* comparisons with SoTA Transformer-based methods.
