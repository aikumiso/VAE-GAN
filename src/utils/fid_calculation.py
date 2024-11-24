import numpy as np
from scipy.linalg import sqrtm
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input

# Load the InceptionV3 model for FID calculation
inception_model = InceptionV3(include_top=False, pooling='avg', input_shape=(75, 75, 3))

def calculate_fid(real_images, generated_images):
    """
    Computes the Fréchet Inception Distance (FID) between two image datasets.

    Args:
        real_images (numpy.ndarray): Real image dataset.
        generated_images (numpy.ndarray): Generated image dataset.

    Returns:
        float: FID score.
    """
    # Preprocess the images
    real_images = preprocess_input(real_images)
    generated_images = preprocess_input(generated_images)

    # Extract features using InceptionV3
    real_features = inception_model.predict(real_images)
    generated_features = inception_model.predict(generated_images)

    # Calculate mean and covariance matrices
    mu1, sigma1 = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = np.mean(generated_features, axis=0), np.cov(generated_features, rowvar=False)

    # Compute FID score
    ssdiff = np.sum((mu1 - mu2) ** 2)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
