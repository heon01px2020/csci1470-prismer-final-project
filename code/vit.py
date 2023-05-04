import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape, LayerNormalization, MultiHeadAttention, Add, Dropout, Input, Embedding, Concatenate, Layer
from tensorflow.math import exp, sqrt, square
import numpy as np


class VIT(tf.keras.Model):
    def __init__(self, args):
        super(VIT, self).__init__()
        self.input_size = (args.num_patches, args.patch_size*args.patch_size*args.num_channels)
        self.dropout_rate = args.dropout_rate
        self.num_patches = args.num_patches
        self.hidden_dim = args.hidden_dim  # H_d
        self.num_layers = args.num_layers
        self.batch_size = args.batch_size
        self.num_classes = args.num_classes
        self.learning_rate = args.learning_rate
        self.loss_list = [] # Append losses to this list in training so you can visualize loss vs time in main
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = self.learning_rate)


        self.patch_embed = Dense(args.hidden_dim)
        self.pos_embed = Embedding(input_dim=self.num_patches, output_dim=self.hidden_dim) ## (256, 768)


        ## CLASSTOKEN LAYER
        self.classtoken = ClassToken()
        ## CLASSTOKEN LAYER END

        ## TRANSFORMER ENCODER LAYERS
        self.trans_enc_layerNorm1 = LayerNormalization()
        self.trans_enc_mhAtten = MultiHeadAttention(num_heads=args.num_heads, key_dim=args.hidden_dim)
        
        self.trans_enc_layerNorm2 = LayerNormalization()
        self.trans_enc_d1 = Dense(args.mlp_dim, activation='gelu')
        self.trans_enc_d2 = Dense(self.hidden_dim, activation='linear')
        ## TRANSFORMER ENCODER END


        ## CLASSIFICATION HEAD LAYERS
        self.class_head_layerNorm = LayerNormalization() ## (None, 257, 768)
        self.class_head_dense = Dense(self.num_classes, activation="linear")
        ## CLASSIFICATION HEAD LAYERS END


        

        ############################################################################################
        #                                      END OF YOUR CODE                                    #
        ############################################################################################

    def call(self, x):
        target_shape = (self.input_size[0], self.input_size[1])
        inputs = tf.keras.layers.Reshape(target_shape)(x)

        """ Patch + Position Embeddings """
        patch_embed = self.patch_embed(inputs)

        positions = tf.range(start=0, limit=self.num_patches, delta=1) ## (256,)
        pos_embed = self.pos_embed(positions) ## (256, 768)
        embed = patch_embed + pos_embed ## (None, 256, 768)
        """ Adding Class Token """
        token = self.classtoken(embed)
        x = Concatenate(axis=1)([token, embed]) ## (None, 257, 768)

        """ Transformer Encoder """
        for _ in range(self.num_layers):
            skip_1 = x
            x = self.trans_enc_layerNorm1(x)
            x = self.trans_enc_mhAtten(x, x)
            x = Add()([x, skip_1])

            skip_2 = x
            x = self.trans_enc_layerNorm2(x)
            x = self.trans_enc_d1(x)
            x = Dropout(self.dropout_rate)(x)
            x = self.trans_enc_d2(x)
            x = Dropout(self.dropout_rate)(x)

            x = Add()([x, skip_2])

        """ Classification Head """
        x = self.class_head_layerNorm(x) ## (None, 257, 768)
        x = x[:, 0, :] ## (None, 768)
        x = Dropout(0.1)(x)
        x = self.class_head_dense(x)
        return x

        ############################################################################################
        #                                      END OF YOUR CODE                                    #
        ############################################################################################

    def loss(self, logits, labels):
        """
        Calculates the model cross-entropy loss after one forward pass.
        Softmax is applied in this function.
        
        :param logits: during training, a matrix of shape (batch_size, self.num_classes) 
        containing the result of multiple convolution and feed forward layers
        :param labels: during training, matrix of shape (batch_size, self.num_classes) containing the train labels
        :return: the loss of the model as a Tensor
        """
        ################ CHECK BACK WHETHER IT SHOULD BE tf.nn.softmax__
        return tf.keras.losses.BinaryCrossentropy(from_logits=True)(labels, logits)
        # return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels, logits))

    def accuracy(self, logits, labels):
        """
        Calculates the model's prediction accuracy by comparing
        logits to correct labels – no need to modify this.
        
        :param logits: a matrix of size (num_inputs, self.num_classes); during training, this will be (batch_size, self.num_classes)
        containing the result of multiple convolution and feed forward layers
        :param labels: matrix of size (num_labels, self.num_classes) containing the answers, during training, this will be (batch_size, self.num_classes)

        NOTE: DO NOT EDIT
        
        :return: the accuracy of the model as a Tensor
        """
        correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    


class ClassToken(Layer):
    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value = w_init(shape=(1, 1, input_shape[-1]), dtype=tf.float32),
            trainable = True
        )

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        hidden_dim = self.w.shape[-1]

        cls = tf.broadcast_to(self.w, [batch_size, 1, hidden_dim])
        cls = tf.cast(cls, dtype=inputs.dtype)
        return cls
    
class Patches(Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

class PatchEncoder(Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = Dense(units=projection_dim)
        self.position_embedding = Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded