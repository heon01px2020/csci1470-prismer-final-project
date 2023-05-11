import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow.keras.layers import Dense, LayerNormalization, MultiHeadAttention, Add, Dropout, Embedding, Concatenate, Layer, Flatten, Reshape


class VIT(tf.keras.Model):
    def __init__(self, args):
        super(VIT, self).__init__()
        self.patch_len = args.patch_size*args.patch_size*args.num_channels
        self.dropout_rate = args.dropout_rate
        self.num_patches = args.num_patches
        self.num_heads = args.num_heads
        self.mlp_dim = args.mlp_dim
        self.latent_dim = args.latent_dim
        self.num_layers = args.num_layers
        self.batch_size = args.batch_size
        self.num_classes = args.num_classes
        self.patch_size = args.patch_size
        self.loss_list = [] # Append losses to this list in training so you can visualize loss vs time in main
        initial_learning_rate = args.learning_rate
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=50,
            decay_rate=0.9,
            staircase=True)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        self.data_augmentation = tf.keras.Sequential(
            [
                tf.keras.layers.Normalization(),
                tf.keras.layers.Resizing(32, 32),
                tf.keras.layers.RandomFlip("horizontal"),
            ],
            name="data_augmentation",)

        self.patch_embed = Dense(self.latent_dim)
        self.pos_embed = Embedding(input_dim=self.num_patches, output_dim=self.latent_dim)


        ## CLASSTOKEN LAYER
        self.classtoken = ClassToken()
        # w_init = tf.keras.initializers.GlorotNormal()
        # self.w = tf.Variable(
        #     initial_value = w_init(shape=(1, 1, self.latent_dim), dtype=tf.float32),
        #     trainable = True
        # )
        ## CLASSTOKEN LAYER END

        ## TRANSFORMER ENCODER LAYERS
        self.trans_enc_layerNorm1 = LayerNormalization()
        self.trans_enc_mhAtten = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.latent_dim)
        
        self.trans_enc_layerNorm2 = LayerNormalization()
        self.trans_enc_d1 = Dense(self.mlp_dim, activation='gelu')
        self.trans_enc_d2 = Dense(self.latent_dim, activation='linear')
        ## TRANSFORMER ENCODER END


        ## CLASSIFICATION HEAD LAYERS
        self.class_head_layerNorm = LayerNormalization()
        self.class_head_dense = Dense(self.num_classes, activation="linear")
        ## CLASSIFICATION HEAD LAYERS END

        ############################################################################################
        #                                      END OF YOUR CODE                                    #
        ############################################################################################

    def call(self, images):
        images = self.data_augmentation(images)

        # Getting patches 
        # (batch_size, img_size, img_size, num_channels) => (batch_size, num_patches, patch_size * patch_size * num_channels)
        patches = tf.image.extract_patches(images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID")
        # print(f"Patches shape: {patches.shape}")
        # print(f"Want to reshape to: {images.shape[0], -1, self.patch_len}")
        patches = Reshape([-1, self.patch_len])(patches)


        # Getting patch embeddings
        # "Transformer uses constant latent vector size D through all of its layers, 
        # so we flatten the patches and map to D dimensions with a trainable linear projection"
        patch_embed = self.patch_embed(patches)

        # Getting positional embeddings
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        pos_embed = self.pos_embed(positions)

        # Adding patch embedding to positional embedding
        embedding = patch_embed + pos_embed
        
        # Adding class token
        class_token = self.classtoken(embedding)

        # class_token = tf.broadcast_to(self.w, [embedding.shape[0], 1, self.w.shape[-1]])
        # class_token = tf.cast(class_token, dtype=embedding.dtype)

        embedding = Concatenate(axis=1)([class_token, embedding])


        # Transformer Encoder
        # "alternating layers of multiheaded selfattention (MSA, see Appendix A) and MLP blocks"
        for _ in range(self.num_layers):
            # "Layernorm (LN) is applied before every block, and residual connections after every block"
            residual = embedding
            embedding = self.trans_enc_layerNorm1(embedding)
            embedding = self.trans_enc_mhAtten(embedding, embedding)
            embedding = Add()([embedding, residual])

            residual = embedding
            embedding = self.trans_enc_layerNorm2(embedding)
            embedding = self.trans_enc_d1(embedding)
            embedding = Dropout(self.dropout_rate)(embedding)
            embedding = self.trans_enc_d2(embedding)
            embedding = Dropout(self.dropout_rate)(embedding)
            embedding = Add()([embedding, residual])

        # MLP Head
        embedding = self.class_head_layerNorm(embedding)
        embedding = Flatten()(embedding)
        # x = x[:, 0, :] ## (None, 768)
        embedding = Dropout(self.dropout_rate)(embedding)

        # Getting logits
        logits = self.class_head_dense(embedding)
        return logits

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
        return tf.keras.losses.CategoricalCrossentropy(from_logits=True)(labels, logits)

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
        labels = tf.cast(labels, tf.int32)
        labels = tf.math.argmax(labels, 1)
        correct_predictions = tf.math.in_top_k(labels, logits, 5)
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    


class ClassToken(Layer):
    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        w_init = tf.keras.initializers.GlorotNormal()
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