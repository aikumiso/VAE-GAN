# Creative AI: Fusion of Transformers with VAEs/GANs

## Overview
This project focuses on designing and implementing a hybrid generative model that combines Variational Autoencoders (VAEs), Generative Adversarial Networks (GANs), and Transformer architectures. The model is applied to creative tasks, such as generating images, with the potential for extensions to other domains like music or text.

The model is evaluated using quantitative metrics such as Fréchet Inception Distance (FID) and qualitative analysis of generated outputs.

## Key Features
- **Hybrid Architecture**: Combines VAEs for structured latent space, GANs for high-quality data generation, and Transformers for handling complex structures.
- **Dataset**: Trained on CIFAR-10 dataset, which includes 60,000 images across 10 categories.
- **Metrics**: Includes Fréchet Inception Distance (FID) for evaluating the quality of generated images.
- **Visualization**: Original test images and generatedimages are compared visually for qualitative assessment.

---

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/aikumiso/VAE-GAN.git
   cd aikumiso
