# Creative AI: Fusion of Transformers with VAEs/GANs

# Overview

This project explores the integration of Variational Autoencoders (VAEs), Generative Adversarial Networks (GANs), and Transformer architectures to enhance creative data generation tasks, such as image synthesis. By combining the structured latent space of VAEs, the high-quality data generation of GANs, and the sequence modeling capabilities of Transformers, the model aims to produce more coherent and realistic outputs.

# Key Features

	•	Hybrid Architecture: Merges VAEs for latent space regularization, GANs for realistic data generation, and Transformers for handling complex data structures.
	•	Dataset: Utilizes the CIFAR-10 dataset, comprising 60,000 images across 10 categories, for training and evaluation.
	•	Evaluation Metrics: Implements the Fréchet Inception Distance (FID) to assess the quality of generated images.
	•	Visualization: Provides side-by-side comparisons of original and generated images for qualitative analysis.

# Installation

	1.	Clone the Repository:

git clone https://github.com/aikumiso/VAE-GAN.git
cd VAE-GAN


	2.	Set Up Environment:
Ensure you have Python 3.7 or higher installed. It’s recommended to use a virtual environment:

python -m venv venv
source venv/bin/activate  # On Windows, use venv\Scripts\activate


	3.	Install Dependencies:

pip install -r requirements.txt

The main dependencies include:
	•	TensorFlow >= 2.0
	•	Keras
	•	numpy
	•	scipy
	•	matplotlib

# How to Run

1. Train the Model

To train the hybrid model, execute:

python train.py

You can adjust training parameters (e.g., batch size, epochs) in the train.py script.

2. Generate Images

After training, generate images by running:

python generate_images.py

This script will produce images using the trained model and save them in the outputs/ directory.

3. Calculate FID Score

To evaluate the quality of the generated images using the FID metric:

python calculate_fid.py

Ensure that the generated images and real images are properly organized as specified in the script.

# Project Structure

├── data/                  # Dataset folder (auto-downloaded CIFAR-10)
├── models/                # Model architectures
│   ├── encoder.py         # Encoder (VAE component)
│   ├── decoder.py         # Decoder (GAN component)
│   └── transformer.py     # Transformer integration
├── utils/                 # Utility functions
│   ├── fid_calculation.py # FID score calculation
│   ├── data_processing.py # Preprocessing and data handling
│   └── visualization.py   # Visualization utilities
├── outputs/               # Generated images and logs
├── train.py               # Training script
├── generate_images.py     # Script to generate images
├── calculate_fid.py       # Script to calculate FID score
├── requirements.txt       # Python package requirements
└── README.md              # Project documentation

# Results

Quantitative Evaluation

	•	FID Score: The model achieved a Fréchet Inception Distance (FID) score of 0.0065, indicating a high similarity between real and generated image distributions.

# Qualitative Analysis

The generated images closely resemble the original dataset in terms of structure and detail. Side-by-side comparisons of original and generated images are included in the visual analysis.

# Visualization

Here is an example of the generated images compared to the original test images:

# Future Work

	1.	Model Improvement:
	•	Increase the latent space size for more detailed image generation.
	•	Experiment with alternative loss functions for better convergence.
	2.	Dataset Expansion:
	•	Train on larger and more complex datasets to improve robustness.
	3.	Extension to Other Domains:
	•	Adapt the model for music and text generation tasks.
	•	Evaluate the framework on multi-modal datasets.

# Citation

If you use this code in your research or projects, please cite this repository:

@misc{CreativeAI,
  author = {Your Name},
  title = {Creative AI: Fusion of Transformers with VAEs/GANs},
  year = {2024},
  url = {https://github.com/aikumiso/VAE-GAN}
}
