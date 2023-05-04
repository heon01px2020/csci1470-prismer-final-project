from __future__ import absolute_import
import tensorflow as tf
import os
import numpy as np
from vit import VIT
from vit2 import ViT
import argparse
from preprocess import get_data
from matplotlib import pyplot as plt
from preprocess import get_data
import math
import cv2

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_weights", action="store_true")
    parser.add_argument("--continue_train", action="store_true")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=100)
    parser.add_argument("--num_patches", type=int, default=16)
    parser.add_argument("--patch_size", type=int, default=8)
    parser.add_argument("--num_channels", type=int, default=3)
    parser.add_argument("--dropout_rate", type=int, default=0.1)
    parser.add_argument("--num_heads", type=int, default=3)
    parser.add_argument("--mlp_dim", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument('--chkpt_path',     default='', help='where the model checkpoint is')
    parser.add_argument('--image_path', type=str, default='')
    args = parser.parse_args()
    return args


# ensures that we run only on cpu
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def save_model(model, args):
    '''Loads model based on arguments'''
    tf.keras.models.save_model(model, args.chkpt_path)
    print(f"Model saved to '{args.chkpt_path}'")

def train(model, train_inputs, train_labels):
    '''
    Trains the model on all of the inputs and labels for one epoch. You should shuffle your inputs 
    and labels - ensure that they are shuffled in the same order using tf.gather or zipping.
    To increase accuracy, you may want to use tf.image.random_flip_left_right on your
    inputs before doing the forward pass. You should batch your inputs.
    
    :param model: the initialized model to use for the forward pass and backward pass
    :param train_inputs: train inputs (all inputs to use for training), 
    shape (num_inputs, width, height, num_channels)
    :param train_labels: train labels (all labels to use for training), 
    shape (num_labels, num_classes)
    :return: Optionally list of losses per batch to use for visualize_loss
    '''

    random_indices = tf.random.shuffle(tf.range(train_inputs.shape[0]))
    # alternatively, considering zipping
    train_inputs = tf.gather(train_inputs, random_indices)
    train_labels = tf.gather(train_labels, random_indices)

    for i in range(math.ceil(train_inputs.shape[0]/model.batch_size)):
        batch = train_inputs[i * model.batch_size:(i+1)*model.batch_size]
        batch = tf.image.random_flip_left_right(batch)
        # Choosing our optimizer
        # Implement backprop:
        with tf.GradientTape() as tape:
            y_pred = model(batch) # this calls the call function conveniently
            loss = model.loss(y_pred, train_labels[i * model.batch_size: (i+1)*model.batch_size])
            # print(loss, model.accuracy(y_pred, train_labels[i * model.batch_size: (i+1)*model.batch_size]))
        model.loss_list.append(loss)
        
        # The keras Model class has the computed property trainable_variables to conveniently
        # return all the trainable variables you'd want to adjust based on the gradients
        ## WHAT WE WANT: tf.Tensor(0.6902424, shape=(), dtype=float32)
        ## WHAT WE HAVE AT HOME: KerasTensor(type_spec=TensorSpec(shape=(), dtype=tf.float32, name=None), 
        ## name='tf.math.reduce_mean/Mean:0', description="created by layer 'tf.math.reduce_mean'")
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return model.loss_list

def test(model, test_inputs, test_labels):
    """
    Tests the model on the test inputs and labels. You should NOT randomly 
    flip images or do any extra preprocessing.
    
    :param test_inputs: test data (all images to be tested), 
    shape (num_inputs, width, height, num_channels)
    :param test_labels: test labels (all corresponding labels),
    shape (num_labels, num_classes)
    :return: test accuracy - this should be the average accuracy across
    all batches
    """
    acc = np.zeros(math.ceil(test_inputs.shape[0]/model.batch_size))
    for i in range(math.ceil(test_inputs.shape[0]/model.batch_size)):
        batch = test_inputs[i * model.batch_size:(i+1)*model.batch_size]
        # Choosing our optimizer
        # Implement backprop:
        y_pred = model(batch) # this calls the call function conveniently
        labels_batch = test_labels[i * model.batch_size: (i+1)*model.batch_size]
        acc[i] = model.accuracy(y_pred, labels_batch)
    return np.mean(acc)


def visualize_loss(losses): 
    """
    Uses Matplotlib to visualize the losses of our model.
    :param losses: list of loss data stored from train. Can use the model's loss_list 
    field 

    NOTE: DO NOT EDIT

    :return: doesn't return anything, a plot should pop-up 
    """
    x = [i for i in range(len(losses))]
    plt.plot(x, losses)
    plt.title('Loss per batch')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.show()  


def visualize_results(image_inputs, probabilities, image_labels, first_label, second_label):
    """
    Uses Matplotlib to visualize the correct and incorrect results of our model.
    :param image_inputs: image data from get_data(), limited to 50 images, shape (50, 32, 32, 3)
    :param probabilities: the output of model.call(), shape (50, num_classes)
    :param image_labels: the labels from get_data(), shape (50, num_classes)
    :param first_label: the name of the first class, "cat"
    :param second_label: the name of the second class, "dog"

    NOTE: DO NOT EDIT

    :return: doesn't return anything, two plots should pop-up, one for correct results,
    one for incorrect results
    """
    # Helper function to plot images into 10 columns
    def plotter(image_indices, label): 
        nc = 10
        nr = math.ceil(len(image_indices) / 10)
        fig = plt.figure()
        fig.suptitle("{} Examples\nPL = Predicted Label\nAL = Actual Label".format(label))
        for i in range(len(image_indices)):
            ind = image_indices[i]
            ax = fig.add_subplot(nr, nc, i+1)
            ax.imshow(image_inputs[ind], cmap="Greys")
            pl = first_label if predicted_labels[ind] == 0.0 else second_label
            al = first_label if np.argmax(
                image_labels[ind], axis=0) == 0 else second_label
            ax.set(title="PL: {}\nAL: {}".format(pl, al))
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.tick_params(axis='both', which='both', length=0)
        
    predicted_labels = np.argmax(probabilities, axis=1)
    num_images = image_inputs.shape[0]

    # Separate correct and incorrect images
    correct = []
    incorrect = []
    for i in range(num_images): 
        if predicted_labels[i] == np.argmax(image_labels[i], axis=0): 
            correct.append(i)
        else: 
            incorrect.append(i)

    plotter(correct, 'Correct')
    plotter(incorrect, 'Incorrect')
    plt.show()

def load_weights(model):
    """
    Load the trained model's weights.

    Inputs:
    - model: Your untrained model instance.
    
    Returns:
    - model: Trained model.
    """
    inputs = tf.zeros([1, 32, 32, args.num_channels])  # Random data sample
    weights_path = os.path.join("model_ckpts", "vit")
    _ = model(inputs)
    model.load_weights(weights_path).expect_partial()
    return model

def train_helper(model, train_inputs, train_labels, args):
    for i in range(args.num_epochs):

        losses = train(model, train_inputs, train_labels)
        if i % 1 == 0:
            train_acc = model.accuracy(model(train_inputs), train_labels)
            print(f"Accuracy on training set after {i+1} training steps: {train_acc}")

    if args.chkpt_path: 
        ## Save model to run testing task afterwards
        save_model(model, args)
    return losses

def main(args):
    '''
    Read in CIFAR10 data (limited to 2 classes), initialize your model, and train and 
    test your model for a number of epochs. We recommend that you train for
    10 epochs and at most 25 epochs. 
    
    CS1470 students should receive a final accuracy 
    on the testing examples for cat and dog of >=70%.
    
    CS2470 students should receive a final accuracy 
    on the testing examples for cat and dog of >=75%.
    
    :return: None
    '''
    train_inputs, train_labels = get_data("../data/train", 3, 5)
    test_inputs, test_labels = get_data("../data/test", 3, 5)

    # Instantiate our model
    model = VIT(args)

    if args.load_weights:
       model = load_weights(model)
       if args.continue_train:
           losses = train_helper(model, train_inputs, train_labels, args)

    else:
        losses = train_helper(model, train_inputs, train_labels, args)

    print()
    test_acc = test(model, test_inputs, test_labels)
    print(f"Accuracy on testing set: {test_acc}")
    model.summary()

    if not args.load_weights:
        visualize_loss(losses)
    if args.image_path:
        img = cv2.imread(args.image_path)
        img = cv2.resize(img, dsize=(32, 32), interpolation=cv2.INTER_CUBIC)
        img = np.reshape(img, (1, 32, 32, 3))
        prob = tf.nn.softmax(model(img))
        pred = tf.math.argmax(prob, axis=-1)
        print("Probability: ", str(prob))
        print("Predicted class (0 is cat, 1 is dog): ", str(pred))
    visualize_results(test_inputs[:50], model(test_inputs)[:50], test_labels[:50], "cat", "dog")
    return


if __name__ == "__main__":
    args = parseArguments()
    main(args)



# def one_hot(labels, class_size):
#     """
#     Create one hot label matrix of size (N, C)

#     Inputs:
#     - labels: Labels Tensor of shape (N,) representing a ground-truth label
#     for each MNIST image
#     - class_size: Scalar representing of target classes our dataset 
#     Returns:
#     - targets: One-hot label matrix of (N, C), where targets[i, j] = 1 when 
#     the ground truth label for image i is j, and targets[i, :j] & 
#     targets[i, j + 1:] are equal to 0
#     """
#     targets = np.zeros((labels.shape[0], class_size))
#     for i, label in enumerate(labels):
#         targets[i, label] = 1
#     targets = tf.convert_to_tensor(targets)
#     targets = tf.cast(targets, tf.float32)
#     return targets

# inner_model = CVAE(28*28, latent_size=15)
# checkpoint_path = "model_ckpts/cvae/cvae"

# num_classes = 10
# inputs = tf.zeros([1, 1, 28, 28])  # Random data sample
# labels = tf.constant([[0]])

# one_hot_vec = one_hot(labels, num_classes)
# print(one_hot_vec)
# _ = inner_model(inputs, one_hot_vec)
# inner_model.load_weights(checkpoint_path)

# inner_model.trainable = False

# model = tf.keras.Sequential(
#     [
#         tf.keras.layers.Dense(32, activation="relu"),
#         inner_model,
#         tf.keras.layers.Dense(1, activation="sigmoid")
#     ]
# )


# show_cvae_images(inner_model, 15)