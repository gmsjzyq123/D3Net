# Domain-Adversarial Generative and Dual Feature Representation Discriminative Network for Hyperspectral Image Domain Generalization.

[Domain-Adversarial Generative and Dual Feature Representation Discriminative Network for Hyperspectral Image Domain Generalization](https://ieeexplore.ieee.org/document/10695100 "Visit OpenAI's official website")
![D3net](figure/D3net.jpg)

## Abstract

Traditional training models often experience significant performance drops on test sets when the training and testing data distributions differ. To address this domain shift problem, we propose D3Net, a domain generalization (DG) classification network for hyperspectral images (HSIs) based on generative adversarial networks (GANs). Specifically, D3Net consists mainly of a generator and a discriminator. The generator extracts domain-invariant information from the source domain to generate data with core classification features, while the discriminator employs a dual-confidence model to enhance the capture of domain-invariant features. Through adversarial iterations, the model is able to adapt to the domain shift effects of unknown data. Unlike existing DG methods that rely on random perturbations for data augmentation, D3Net utilizes learnable convolutional neural networks (CNNs) rather than randomization to enhance the model’s learning capability.We conducted cross-scene classification experiments on datasets from Houston, Pavia, and Indiana, and the results demonstrate the effectiveness of our approach. The code for D3Net is available at: https://github.com/gmsjzyq123/D3Net.

## Citation
```
@ARTICLE{10695100,
  author={Chu, Minghui and Yu, Xiaodong and Dong, Hongbin and Zang, Shuying},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Domain-Adversarial Generative and Dual-Feature Representation Discriminative Network for Hyperspectral Image Domain Generalization}, 
  year={2024},
  volume={62},
  number={},
  pages={1-13},
  keywords={Hyperspectral imaging;Data models;Feature extraction;Generators;Training;Generative adversarial networks;Adaptation models;Contrastive learning;domain generalization (DG);generative adversarial network (GAN);hyperspectral image (HSI) classification},
  doi={10.1109/TGRS.2024.3468311}}
```

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
   
