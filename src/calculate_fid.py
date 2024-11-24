import numpy as np
from utils.fid_calculation import calculate_fid
from utils.data_processing import load_data, resize_images
from tensorflow.keras.models import load_model

# Load data
x_train, x_test = load_data()

# Resize images to 75x75 for InceptionV3
real_images = resize_images(x_test[:100])
vae_gan = load_model("outputs/models/vae_gan.h5")
decoder = vae_gan.get_layer("decoder")

# Generate images
latent_space = np.random.normal(size=(100, 128))
generated_images = decoder.predict(latent_space)
generated_images = resize_images(generated_images)

# Calculate FID score
fid_score = calculate_fid(real_images, generated_images)
print(f"FID Score: {fid_score}")
