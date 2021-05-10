# MLP Mixer

PyTorch implementation of [MLP-Mixer: An all-MLP Architecture for Vision](https://arxiv.org/abs/2105.01601).

![alt text](https://miro.medium.com/max/3398/1*cUHd6G9jjwl9F7xXv_SBEw.jpeg)

## Quickstart

Clone this repository.

```
git clone https://github.com/jaketae/mlp-mixer.git
```

Navigate to the cloned directory. You can start using the model via

```python
>>> from mlp_mixer import MLPMixer
>>> model = MLPMixer()
```

By default, the model comes with the following parameters:

```python
MLPMixer(
    image_size=256,
    patch_size=16,
    in_channels=3,
    num_features=128,
    expansion_factor=2,
    num_layers=8,
    num_classes=10,
    dropout=0.5,
)
```

## Summary

Convolutional Neural Networks (CNNs) and transformers are two mainstream model architectures currently dominating computer vision and natural language processing. The authors of the paper, however, empirically show that neither convolution nor self-attenion are necessary; in fact, muti-layered perceptrons (MLPs) can also serve as a strong baseline. The authors present MLP-Mixer, an all-MLP mode architecture, that contains two types of layers: a token-mixing layer and a channel-mixing layer. Each of the layers "mix" per-location and per-feature information in the input. MLP-Mixer performs comparably to other state-of-the-art models, such as [ViT](https://arxiv.org/abs/2010.11929) or [EfficientNet](https://arxiv.org/abs/1905.11946).

## Resources

- [Original Paper](https://arxiv.org/abs/2105.01601)
- [Phil Wang's implementation](https://github.com/lucidrains/mlp-mixer-pytorch)
- [Rishikesh's implementation](https://github.com/rishikksh20/MLP-Mixer-pytorch)
- [Image from Nabil Madali's Medium article](https://medium.com/@nabil.madali/an-all-mlp-architecture-for-vision-7e7e1270fd33)
