import numpy as np
from utils.visualization import plot_images
from tensorflow.keras.models import load_model

# Load trained model
vae_gan = load_model("outputs/models/vae_gan.h5")
decoder = vae_gan.get_layer("decoder")

# Generate random images
latent_space = np.random.normal(size=(9, 128))
generated_images = decoder.predict(latent_space)

# Visualize generated images
plot_images(generated_images, title="Generated Images")
