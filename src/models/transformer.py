import tensorflow as tf
from tensorflow.keras import layers

class TransformerBlock(layers.Layer):
    """
    Transformer block for processing latent vectors or feature maps.
    """
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        """
        Initializes the Transformer block.

        Args:
            embed_dim (int): Dimensionality of the embedding space.
            num_heads (int): Number of attention heads.
            ff_dim (int): Dimensionality of the feed-forward layer.
            rate (float): Dropout rate.
        """
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        """
        Forward pass for the Transformer block.
        """
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


def build_transformer_model(input_shape, num_layers, embed_dim, num_heads, ff_dim, num_classes):
    """
    Builds a Transformer-based model.

    Args:
        input_shape (tuple): Shape of the input data.
        num_layers (int): Number of Transformer blocks.
        embed_dim (int): Dimensionality of the embedding space.
        num_heads (int): Number of attention heads.
        ff_dim (int): Dimensionality of the feed-forward layer.
        num_classes (int): Number of output classes (if classification is needed).

    Returns:
        model (tf.keras.Model): The Transformer-based model.
    """
    inputs = layers.Input(shape=input_shape)
    
    # Flatten the input for the Transformer
    x = layers.Flatten()(inputs)
    x = layers.Dense(embed_dim)(x)  # Initial embedding

    # Add Transformer blocks
    for _ in range(num_layers): 
        x = TransformerBlock(embed_dim, num_heads, ff_dim)(x)

    # Output layer
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


if __name__ == "__main__":
    # Example usage
    input_shape = (32, 32, 3)  # CIFAR-10 image size
    num_layers = 2
    embed_dim = 128
    num_heads = 4
    ff_dim = 256
    num_classes = 10  # For classification tasks

    model = build_transformer_model(input_shape, num_layers, embed_dim, num_heads, ff_dim, num_classes)
    model.summary()
