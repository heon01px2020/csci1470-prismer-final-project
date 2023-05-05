import math
import numpy as np
import tensorflow as tf

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, emb_sz, MultiHead=False, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)

        # TODO:
        # 1) Define the Feed Forward, self-attention, encoder-decoder-attention, and layer normalization layers
        # 2) For 2470 students, use multiheaded attention

        self.ff_layer = tf.keras.layers.Dense(emb_sz, activation="relu") 

        self.self_atten = tf.keras.layers.MultiHeadAttention(num_heads = 2, key_dim = int(emb_sz/2)) #HELP??
        self.self_context_atten = tf.keras.layers.MultiHeadAttention(num_heads = 2, key_dim=int(emb_sz/2))
        #self.self_atten         = AttentionHead(emb_sz, emb_sz, True)  #if not MultiHead else MultiHeadedAttention(emb_sz, True)
        #self.self_context_atten = AttentionHead(emb_sz, emb_sz, False) #if not MultiHead else MultiHeadedAttention(emb_sz, False)
        self.layer_norm = tf.keras.layers.LayerNormalization() 

    @tf.function
    def call(self, inputs, context_sequence):
        """
        This functions calls a transformer block.

        TODO:
        1) compute MASKED attention on the inputs
        2) residual connection and layer normalization
        3) computed UNMASKED attention using context
        4) residual connection and layer normalization
        5) feed forward layer
        6) residual layer and layer normalization
        7) return relu of tensor

        NOTES: This article may be of great use:
        https://www.tensorflow.org/text/tutorials/transformer#the_embedding_and_positional_encoding_layer

        :param inputs: tensor of shape [BATCH_SIZE x INPUT_SEQ_LENGTH x EMBEDDING_SIZE ]
        :param context_sequence: tensor of shape [BATCH_SIZE x CONTEXT_SEQ_LENGTH x EMBEDDING_SIZE ]
        :return: tensor of shape [BATCH_SIZE x INPUT_SEQ_LENGTH x EMBEDDING_SIZE ]
        """
        # masked_atten_inputs = self.self_atten(inputs, inputs, inputs)
        masked_atten_inputs = self.self_atten(inputs, inputs)
        masked_atten_inputs = masked_atten_inputs + inputs
        masked_atten_inputs = self.layer_norm(masked_atten_inputs)
        # unmasked_atten_context = self.self_context_atten(context_sequence, context_sequence, masked_atten_inputs) 
        unmasked_atten_context = self.self_context_atten(context_sequence, masked_atten_inputs) 
        # residual connection and layer normalization:
        unmasked_atten_context = unmasked_atten_context + masked_atten_inputs
        unmasked_atten_context = self.layer_norm(unmasked_atten_context)
        ff_output = self.ff_layer(unmasked_atten_context)
        # residual connection and layer normalization:
        ff_output = ff_output + unmasked_atten_context
        ff_output = self.layer_norm(ff_output)

        output = tf.nn.relu(ff_output)
        #output  = tf.keras.layers.Softmax()
        
        return output


def positional_encoding(length, depth):
    ## REFERENCE: https://www.tensorflow.org/text/tutorials/transformer#the_embedding_and_positional_encoding_layer
    depth = depth/2
    ## Generate a range of positions and depths 
    positions = np.arange(length)[:, np.newaxis]    # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :]/depth  # (1, depth)
    ## Compute range of radians to take the sine and cosine of.
    angle_rates = 1 / (10000**depths)               # (1, depth)
    angle_rads = positions * angle_rates            # (pos, depth)
    pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1) 
    ## This serves as offset for the Positional Encoding
    return tf.cast(pos_encoding, dtype=tf.float32)


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embed_size, window_size):
        super().__init__()
        self.embed_size = embed_size

        ## TODO: Implement Component

        ## Embed labels into an optimizable embedding space
        self.embedding = tf.keras.layers.Embedding(vocab_size, embed_size, mask_zero=True) #input_length = 2?

        ## Implement sinosoidal positional encoding: offset by varying sinosoidal frequencies. 
        ## HINT: May want to use the function above...
        self.pos_encoding = positional_encoding(window_size, embed_size)

    def call(self, x):
        ## TODO: Get embeddings and and scale them by sqrt of embedding size, and add positional encoding.
        embedded_inputs = self.embedding(x)
        scaled_embedded_inputs = embedded_inputs * (math.sqrt(self.embed_size))
        pos_encoded = scaled_embedded_inputs + self.pos_encoding#[:tf.shape(embedded_inputs)[1], :] #IDK ABOUT THIS
        return pos_encoded