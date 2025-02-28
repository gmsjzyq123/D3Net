# Domain-Adversarial Generative and Dual Feature Representation Discriminative Network for Hyperspectral Image Domain Generalization.

![D3net](figure/D3net.jpg)

## Abstract

'''
Traditional training models often experience significant performance drops on test sets when the training and testing data distributions differ. To address this domain shift problem, we propose D3Net, a domain generalization (DG) classification network for hyperspectral images (HSIs) based on generative adversarial networks (GANs). Specifically, D3Net consists mainly of a generator and a discriminator. The generator extracts domain-invariant information from the source domain to generate data with core classification features, while the discriminator employs a dual-confidence model to enhance the capture of domain-invariant features. Through adversarial iterations, the model is able to adapt to the domain shift effects of unknown data. Unlike existing DG methods that rely on random perturbations for data augmentation, D3Net utilizes learnable convolutional neural networks (CNNs) rather than randomization to enhance the model’s learning capability.We conducted cross-scene classification experiments on datasets from Houston, Pavia, and Indiana, and the results demonstrate the effectiveness of our approach. The code for D3Net is available at: https://github.com/gmsjzyq123/D3Net.
'''

## Datasets

```
datasets
├── Houston
│   ├── Houston13.mat
│   ├── Houston13_7gt.mat
│   ├── Houston18.mat
│   └── Houston18_7gt.mat
└── Pavia
│   ├── paviaC.mat
│   └── paviaC_7gt.mat
│   ├── paviaU.mat
│   └── paviaU_7gt.mat
└── Indiana
```
   
