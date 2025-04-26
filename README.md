# GMAN: Generative Multi-Adversarial Networks

## Overview

Generative Multi-Adversarial Networks (GMAN) is a novel GAN framework designed to improve training stability, convergence speed, and generative diversity using **multiple discriminators**. Unlike prior approaches that alter the objective function or introduce additional generators, GMAN keeps the original GAN objective intact and instead varies discriminator architecture and dropout rates.

## Motivation

Traditional GANs often suffer from issues such as:

- Mode collapse
- Unstable training
- Slow convergence

GMAN addresses these challenges by incorporating **1–4 discriminators** with varying dropout and convolutional structures, allowing the generator to receive more diverse and robust feedback during training.

## Key Contributions

- Introduced a multi-discriminator GAN (up to 4 discriminators) with different dropout and internal configurations.
- Retained the **original GAN loss function** to maintain generalizability.
- Demonstrated **improvements in generation quality, training stability**, and **convergence speed** on natural image datasets.

## Research Questions

1. Does using multiple discriminators improve GAN training stability and convergence?
2. Can GMAN enhance generative diversity and sample quality?
3. Is GMAN generalizable across image datasets without modifying the loss function?

## Architecture

- **Generator**: Maps noise vectors to realistic images.
- **Discriminators (1–4)**: Each with a unique configuration, they evaluate the realism of generated samples.
- **Loss**: Binary Cross Entropy (BCE) for both generator and discriminators.

## Experimental Setup

- **Datasets**: MNIST, CelebA, CIFAR-10
- **Train/Test Split**: 80/20
- **Environment**:
  - 3 Windows 11 machines with GPU
  - 64 GB RAM
  - Python 3.10, PyTorch 2.1
- **Hyperparameters**:
  - Latent dimension: 100
  - Optimizer: Adam (lr = 0.002)
  - Batch size: 32
  - Epochs: 10

## Baselines

- **Vanilla GAN**: Standard single discriminator model
- **MAD-GAN**: Multiple generators, one discriminator
- **MGGAN**: Uses auxiliary features for enhanced generation
- **MMGAN**: Uses GMM latent space for multi-modal data generation

## Evaluation Metrics

- **Convergence Speed**: Analyzed using loss curves
- **Training Stability**: Measured by loss variance and quality trends
- **Visual Quality**: Compared across 1–4 discriminators for all datasets

## Results

- **Improved stability** and **faster convergence** observed with multiple discriminators.
- **Enhanced image quality** across MNIST, CIFAR-10, and CelebA.
- Maintained **generalizability** without altering the GAN loss.

## References

1. Ghosh et al., “Multi-Agent Diverse Generative Adversarial Networks,” [arXiv:1704.02906](http://arxiv.org/abs/1704.02906)
2. Yu & Liu, “Multiple Granularities GAN for Wafer Map Defects,” IEEE Transactions, 2022
3. Pandeva & Schubert, “MMGAN: Multi-Modal GANs,” [arXiv:1911.06663](http://arxiv.org/abs/1911.06663)
4. Durugkar et al., “Generative Multi-Adversarial Networks,” [arXiv:1611.01673](http://arxiv.org/abs/1611.01673)

## License

MIT License
