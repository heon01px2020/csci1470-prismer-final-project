import tensorflow as tf
from tensorflow import keras
from transformer import TransformerBlock, PositionalEncoding

class TransformerDecoder(tf.keras.Model):

    def __init__(self, vocab_size, hidden_size, window_size, **kwargs):

        super().__init__(**kwargs)
        self.vocab_size  = vocab_size
        self.hidden_size = hidden_size
        self.window_size = window_size

        # TODO: Define image and positional encoding, transformer decoder, and classification layers

        # Define feed forward layer to embed image features into a vector 
        self.image_embedding = tf.keras.layers.Dense(self.hidden_size, activation="relu")

        # Define positional encoding to embed and offset layer for language:
        self.encoding = PositionalEncoding(self.vocab_size, self.hidden_size, self.window_size)
        #self.encoding = keras_nlp.layers.SinePositionEncoding()

        # Define transformer decoder layer:
        self.decoder = TransformerBlock(self.hidden_size)

        # Define classification layer (logits)
        self.classifier = tf.keras.layers.Dense(self.vocab_size, activation="linear")

    def call(self, encoded_images, questions):
        # TODO:
        # 1) Embed the encoded images into a vector (HINT IN NOTEBOOK)
        # 2) Pass the captions through your positional encoding layer
        # 3) Pass the english embeddings and the image sequences to the decoder
        # 4) Apply dense layer(s) to the decoder out to generate logits
        encoded_images = self.image_embedding(tf.expand_dims(encoded_images, 1))
        questions = self.encoding(questions)
        decoder_output = self.decoder(questions, encoded_images)
        
        probs = self.classifier(decoder_output)
        return probs
    
    # def get_config(self):
    #     return {"vocab_size" : self.vocab_size, "hidden_size" : self.hidden_size, "window_size" : self.window_size}
    
    # @classmethod
    # def from_config(cls, config):
    #     return cls(**config)
