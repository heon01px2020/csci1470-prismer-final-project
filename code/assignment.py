import argparse
import math
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import tensorflow as tf
from tensorflow.math import sigmoid
from tqdm import tqdm

from vae import VAE, CVAE, reparametrize, loss_function


def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_cvae", action="store_true")
    parser.add_argument("--load_weights", action="store_true")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--latent_size", type=int, default=15)
    parser.add_argument("--input_size", type=int, default=28 * 28)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    args = parser.parse_args()
    return args


def one_hot(labels, class_size):
    """
    Create one hot label matrix of size (N, C)

    Inputs:
    - labels: Labels Tensor of shape (N,) representing a ground-truth label
    for each MNIST image
    - class_size: Scalar representing of target classes our dataset 
    Returns:
    - targets: One-hot label matrix of (N, C), where targets[i, j] = 1 when 
    the ground truth label for image i is j, and targets[i, :j] & 
    targets[i, j + 1:] are equal to 0
    """
    targets = np.zeros((labels.shape[0], class_size))
    for i, label in enumerate(labels):
        targets[i, label] = 1
    targets = tf.convert_to_tensor(targets)
    targets = tf.cast(targets, tf.float32)
    return targets


def train_vae(model, train_loader, args, is_cvae=False):
    """
    Train your VAE with one epoch.

    Inputs:
    - model: Your VAE instance.
    - train_loader: A tf.data.Dataset of MNIST dataset.
    - args: All arguments.
    - is_cvae: A boolean flag for Conditional-VAE. If your model is a Conditional-VAE,
    set is_cvae=True. If it's a Vanilla-VAE, set is_cvae=False.

    Returns:
    - total_loss: Sum of loss values of all batches.
    """
    # Use the given one_hot() to convert the image labels, 
    # and pass the images and their one-hot vectors 
    # to the model to get the generated images. 
    # After training for 10 epochs, your loss should be <120.

    optimizer = tf.keras.optimizers.Adam(learning_rate = args.learning_rate)
    total_loss = 0

    for batch in train_loader:
        with tf.GradientTape() as tape:
            if is_cvae:
                batch_hat, mu, logvar = model(x=batch[0], c=one_hot(batch[1],10))
            else:
                batch_hat, mu, logvar = model(batch[0])
            loss = loss_function(batch_hat, batch[0], mu, logvar)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        total_loss += loss

    return total_loss


def load_mnist(batch_size, buffer_size=1024):
    """
    Load and preprocess MNIST dataset from tf.keras.datasets.mnist.

    Inputs:
    - batch_size: An integer value of batch size.
    - buffer_size: Buffer size for random sampling in tf.data.Dataset.shuffle().

    Returns:
    - train_dataset: A tf.data.Dataset instance of MNIST dataset. Batching and shuffling are already supported.
    """
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), _ = mnist.load_data()
    x_train = x_train / 255.0
    x_train = np.expand_dims(x_train, axis=1)  # [batch_sz, channel_sz, height, width]
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=buffer_size).batch(batch_size, drop_remainder=True)
    return train_dataset


def save_model_weights(model, args):
    """
    Save trained VAE model weights to model_ckpts/

    Inputs:
    - model: Trained VAE model.
    - args: All arguments.
    """
    model_flag = "cvae" if args.is_cvae else "vae"
    output_dir = os.path.join("model_ckpts", model_flag)
    output_path = os.path.join(output_dir, model_flag)
    os.makedirs("model_ckpts", exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    model.save_weights(output_path)


def show_vae_images(model, latent_size):
    """
    Call this only if the model is VAE!
    Generate 10 images from random vectors.
    Show the generated images from your trained VAE.
    Image will be saved to outputs/show_vae_images.pdf

    Inputs:
    - model: Your trained model.
    - latent_size: Latent size of your model.
    """
    # Generated images from vectors of random values.
    z = tf.random.normal(shape=[10, latent_size])
    samples = model.decoder(z).numpy()

    # Visualize
    fig = plt.figure(figsize=(10, 1))
    gspec = gridspec.GridSpec(1, 10)
    gspec.update(wspace=0.05, hspace=0.05)
    for i, sample in enumerate(samples):
        ax = plt.subplot(gspec[i])
        plt.axis("off")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect("equal")
        plt.imshow(sample.reshape(28, 28), cmap="Greys_r")

    # Save the generated images
    os.makedirs("outputs", exist_ok=True)
    output_path = os.path.join("outputs", "show_vae_images.pdf")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def show_vae_interpolation(model, latent_size):
    """
    Call this only if the model is VAE!
    Generate interpolation between two .
    Show the generated images from your trained VAE.
    Image will be saved to outputs/show_vae_interpolation.pdf

    Inputs:
    - model: Your trained model.
    - latent_size: Latent size of your model.
    """

    def show_interpolation(images):
        """
        A helper to visualize the interpolation.
        """
        images = tf.reshape(images, [images.shape[0], -1])  # images reshape to (batch_size, D)
        sqrtn = int(math.ceil(math.sqrt(images.shape[0])))
        sqrtimg = int(math.ceil(math.sqrt(images.shape[1])))

        fig = plt.figure(figsize=(sqrtn, sqrtn))
        gs = gridspec.GridSpec(sqrtn, sqrtn)
        gs.update(wspace=0.05, hspace=0.05)
        for i, img in enumerate(images):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(tf.reshape(img, [sqrtimg, sqrtimg]))

        # Save the generated images
        os.makedirs("outputs", exist_ok=True)
        output_path = os.path.join("outputs", "show_vae_interpolation.pdf")
        plt.savefig(output_path, bbox_inches="tight")
        plt.close(fig)

    S = 12
    z0 = tf.random.normal(shape=[S, latent_size], dtype=tf.dtypes.float32)  # [S, latent_size]
    z1 = tf.random.normal(shape=[S, latent_size], dtype=tf.dtypes.float32)
    w = tf.linspace(0, 1, S)
    w = tf.cast(tf.reshape(w, (S, 1, 1)), dtype=tf.float32)  # [S, 1, 1]
    z = tf.transpose(w * z0 + (1 - w) * z1, perm=[1, 0, 2])
    z = tf.reshape(z, (S * S, latent_size))  # [S, S, latent_size]
    x = model.decoder(z)  # [S*S, 1, 28, 28]
    show_interpolation(x)


def show_cvae_images(model, latent_size):
    """
    Call this only if the model is CVAE!
    Conditionally generate 10 images for each digit.
    Show the generated images from your trained CVAE.
    Image will be saved to outputs/show_cvae_images.pdf

    Inputs:
    - model: Your trained model.
    - latent_size: Latent size of your model.
    """
    # Conditionally generated images from vectors of random values.
    num_generation = 100
    num_classes = 10
    num_per_class = num_generation // num_classes
    c = tf.eye(num_classes)  # [one hot labels for 0-9]
    z = []
    labels = []
    for label in range(num_classes):
        curr_c = c[label]
        curr_c = tf.broadcast_to(curr_c, [num_per_class, len(curr_c)])
        curr_z = tf.random.normal(shape=[num_per_class, latent_size])
        curr_z = tf.concat([curr_z, curr_c], axis=-1)
        z.append(curr_z)
        labels.append([label] * num_per_class)
    z = np.concatenate(z)
    labels = np.concatenate(labels)
    samples = model.decoder(z).numpy()

    # Visualize
    rows = num_classes
    cols = num_generation // rows

    fig = plt.figure(figsize=(cols, rows))
    gspec = gridspec.GridSpec(rows, cols)
    gspec.update(wspace=0.05, hspace=0.05)
    for i, sample in enumerate(samples):
        ax = plt.subplot(gspec[i])
        plt.axis("off")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect("equal")
        plt.imshow(sample.reshape(28, 28), cmap="Greys_r")

    # Save the generated images
    os.makedirs("outputs", exist_ok=True)
    output_path = os.path.join("outputs", "show_cvae_images2.pdf")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def load_weights(model):
    """
    Load the trained model's weights.

    Inputs:
    - model: Your untrained model instance.
    
    Returns:
    - model: Trained model.
    """
    num_classes = 10
    inputs = tf.zeros([1, 1, 28, 28])  # Random data sample
    labels = tf.constant([[0]])
    if args.is_cvae:
        weights_path = os.path.join("model_ckpts", "cvae", "cvae")
        one_hot_vec = one_hot(labels, num_classes)
        _ = model(inputs, one_hot_vec)
        model.load_weights(weights_path)
    else:
        weights_path = os.path.join("model_ckpts", "vae", "vae")
        _ = model(inputs)
        model.load_weights(weights_path)
    return model


def main(args):
    # Load MNIST dataset
    train_dataset = load_mnist(args.batch_size)

    # Get an instance of VAE
    args.is_cvae = True
    if args.is_cvae:
        model = CVAE(args.input_size, latent_size=args.latent_size)
    else:
        model = VAE(args.input_size, latent_size=args.latent_size)

    # Load trained weights
    if args.load_weights:
       model = load_weights(model)

    # Train VAE
    for epoch_id in range(args.num_epochs):
        total_loss = train_vae(model, train_dataset, args, is_cvae=args.is_cvae)
        print(f"Train Epoch: {epoch_id} \tLoss: {total_loss / len(train_dataset):.6f}")

    # Visualize results
    if args.is_cvae:
        show_cvae_images(model, args.latent_size)
    else:
        show_vae_images(model, args.latent_size)
        show_vae_interpolation(model, args.latent_size)

    # Optional: Save VAE/CVAE model for debugging/testing.
    save_model_weights(model, args)


if __name__ == "__main__":
    args = parseArguments()
    main(args)
