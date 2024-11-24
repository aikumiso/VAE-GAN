import tensorflow as tf
from tensorflow.keras import layers, models

def build_encoder(input_shape):
    inputs = layers.Input(shape=input_shape)
    
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.Flatten()(x)
    latent = layers.Dense(128, activation='relu')(x)
    
    return models.Model(inputs, latent, name="encoder")
