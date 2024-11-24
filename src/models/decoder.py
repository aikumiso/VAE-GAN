import tensorflow as tf
from tensorflow.keras import layers, models

def build_decoder(output_shape):
    latent_dim = 128
    inputs = layers.Input(shape=(latent_dim,))
    
    x = layers.Dense(8 * 8 * 128, activation='relu')(inputs)
    x = layers.Reshape((8, 8, 128))(x)
    x = layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    outputs = layers.Conv2DTranspose(output_shape[-1], (3, 3), activation='sigmoid', padding='same')(x)
    
    return models.Model(inputs, outputs, name="decoder")
