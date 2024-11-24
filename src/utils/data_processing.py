import tensorflow as tf

def load_data():
    """
    Loads and preprocesses the CIFAR-10 dataset.

    Returns:
        x_train (numpy.ndarray): Preprocessed training data, normalized to [0, 1].
        x_test (numpy.ndarray): Preprocessed testing data, normalized to [0, 1].
    """
    (x_train, _), (x_test, _) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    return x_train, x_test

def resize_images(images, target_size=(75, 75)):
    """
    Resizes a batch of images to the target size.

    Args:
        images (numpy.ndarray): Input images.
        target_size (tuple): Target size (height, width).

    Returns:
        numpy.ndarray: Resized images.
    """
    resized_images = tf.image.resize(images, target_size).numpy()
    return resized_images
