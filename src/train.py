from utils.data_processing import load_data
from src.models.vae_gan import build_vae_gan
from tensorflow.keras.callbacks import ModelCheckpoint

# Load data
x_train, x_test = load_data()

# Build model
vae_gan, encoder, decoder = build_vae_gan(input_shape=x_train.shape[1:])
vae_gan.compile(optimizer="adam", loss="mse")

# Training
vae_gan.fit(
    x_train, x_train,
    epochs=20,
    batch_size=32,
    validation_data=(x_test, x_test),
    callbacks=[
        ModelCheckpoint(filepath="outputs/models/vae_gan.h5", save_best_only=True)
    ]
)
