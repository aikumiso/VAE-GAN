import tensorflow as tf

def load_data():
    """
    Loads and preprocesses the CIFAR-10 dataset.

    Returns:
        x_train (numpy.ndarray): Preprocessed training data, normalized to [0, 1].
        x_test (numpy.ndarray): Preprocessed testing data, normalized to [0, 1].
    """
    # Load CIFAR-10 dataset
    (x_train, _), (x_test, _) = tf.keras.datasets.cifar10.load_data()

    # Normalize data to range [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    return x_train, x_test

def main():
    """
    Main function to test the data loader.
    """
    # Load the data
    x_train, x_test = load_data()

    print("Training data shape:", x_train.shape)
    print("Testing data shape:", x_test.shape)

if __name__ == "__main__":
    main()
