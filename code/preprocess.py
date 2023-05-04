# preprocess TextVQA dataset

import json
import pickle
import random
import re
from PIL import Image
import tensorflow as tf
import numpy as np
import collections
from tqdm import tqdm

def preprocess_captions(captions, window_size):
    for i, caption in enumerate(captions):
        # Taken from:
        # https://towardsdatascience.com/image-captions-with-attention-in-tensorflow-step-by-step-927dad3569fa

        # Convert the caption to lowercase, and then remove all special characters from it
        caption_nopunct = re.sub(r"[^a-zA-Z0-9]+", ' ', caption.lower())
      
        # Split the caption into separate words, and collect all words which are more than 
        # one character and which contain only alphabets (ie. discard words with mixed alpha-numerics)
        clean_words = [word for word in caption_nopunct.split() if ((len(word) > 1) and (word.isalpha()))]
      
        # Join those words into a string
        caption_new = ['<start>'] + clean_words[:window_size-1] + ['<end>']
      
        # Replace the old caption in the captions list with this new cleaned caption
        captions[i] = caption_new

def get_image_features(image_names, data_folder, vis_subset=100):
    '''
    Method used to extract the features from the images in the dataset using ResNet50
    '''
    image_features = []
    vis_images = []
    resnet = tf.keras.applications.ResNet50(False)  ## Produces Bx7x7x2048
    gap = tf.keras.layers.GlobalAveragePooling2D()  ## Produces Bx2048
    pbar = tqdm(image_names)
    for i, image_name in enumerate(pbar):
        img_path = f'{data_folder}/Images/{image_name}'
        pbar.set_description(f"[({i+1}/{len(image_names)})] Processing '{img_path}' into 2048-D ResNet GAP Vector")
        with Image.open(img_path) as img:
            img_array = np.array(img.resize((224,224)))
        img_in = tf.keras.applications.resnet50.preprocess_input(img_array)[np.newaxis, :]
        image_features += [gap(resnet(img_in))]
        if i < vis_subset:
            vis_images += [img_array]
    print()
    return image_features, vis_images


def load_data(data_folder):
    '''
    Method that was used to preprocess the data in the data.p file. You do not need 
    to use this method, nor is this used anywhere in the assignment. This is the method
    that the TAs used to pre-process the Flickr 8k dataset and create the data.p file 
    that is in your assignment folder. 

    Feel free to ignore this, but please read over this if you want a little more clairity 
    on how the images and captions were pre-processed 


    '''
    train_text_file_path = f'{data_folder}/textvqa_train.json'

    with open(train_text_file_path) as train_file:
        train_data = json.load(train_file) # python dictionary

    # map each image_id to a dictionary containing the question and a list containing all the answers
    train_image_ids_to_questions_and_answers = {}
    for dict in train_data:
        image_id = dict["image_id"]
        questions = dict["question"]
        answers = dict["answers"]
        train_image_ids_to_questions_and_answers[image_id] = {
            "question_tokens": questions,
            "answers": answers
            }
    
    test_text_file_path = f'{data_folder}/textvqa_test.json'

    with open(test_text_file_path) as test_file:
        test_data = json.load(test_file) # python dictionary
    
    test_image_ids_to_questions_and_answers = {}
    for dict in test_data:
        image_id = dict["image_id"]
        questions = dict["question"]
        answers = dict["answers"]
        test_image_ids_to_questions_and_answers[image_id] = {
            "question_tokens": questions,
            "answers": answers
            }
    

    # #map each image name to a list containing all 5 of its captons
    # image_names_to_captions = {}
    # for example in examples:
    #     img_name, caption = example.split(',', 1)
    #     image_names_to_captions[img_name] = image_names_to_captions.get(img_name, []) + [caption]

    #randomly split examples into training and testing sets
    # shuffled_images = list(image_names_to_captions.keys())
    # random.seed(0)
    # random.shuffle(shuffled_images)
    # test_image_names = shuffled_images[:1000]
    # train_image_names = shuffled_images[1000:]

    def get_all_questions(image_ids_to_questions_and_answers):
        to_return = []
        for image_id in image_ids_to_questions_and_answers:
            question = image_ids_to_questions_and_answers[image_id]["question"]
            to_return.append(question)
        return to_return


    # get lists of all the questions in the train and testing set
    train_questions = get_all_questions(train_image_ids_to_questions_and_answers)
    test_questions = get_all_questions(test_image_ids_to_questions_and_answers)


    def get_all_answers(image_ids_to_questions_and_answers):
        to_return = []
        for image_id in image_ids_to_questions_and_answers:
            answers = image_ids_to_questions_and_answers[image_id]["answers"]
            to_return.append(answers)
        return to_return


    # get lists of all the ansers in the train and testing set
    train_answers = get_all_answers(train_image_ids_to_questions_and_answers) #a list of lists of answers
    test_answers = get_all_answers(test_image_ids_to_questions_and_answers)

    #remove special charachters and other nessesary preprocessing
    window_size = 20
    preprocess_captions(train_questions, window_size)
    preprocess_captions(test_questions, window_size)

    # count word frequencies and replace rare words with '<unk>'
    word_count = collections.Counter()
    for caption in train_questions:
        word_count.update(caption)

    def unk_captions(captions, minimum_frequency):
        for caption in captions:
            for index, word in enumerate(caption):
                if word_count[word] <= minimum_frequency:
                    caption[index] = '<unk>'

    unk_captions(train_questions, 50)
    unk_captions(test_questions, 50)

    # pad captions so they all have equal length
    def pad_captions(captions, window_size):
        for caption in captions:
            caption += (window_size + 1 - len(caption)) * ['<pad>'] 
    
    pad_captions(train_questions, window_size)
    pad_captions(test_questions,  window_size)

    # assign unique ids to every word left in the vocabulary
    word2idx = {}
    vocab_size = 0
    for question in train_questions:
        for index, word in enumerate(question):
            if word in word2idx:
                question[index] = word2idx[word]
            else:
                word2idx[word] = vocab_size
                question[index] = vocab_size
                vocab_size += 1
    for question in test_questions:
        for index, word in enumerate(question):
            question[index] = word2idx[word] 
    
    # use ResNet50 to extract image features
    train_image_ids = train_image_ids_to_questions_and_answers.keys()
    test_image_ids = test_image_ids_to_questions_and_answers.keys()

    print("Getting training embeddings")
    train_image_features, train_images = get_image_features(train_image_ids, data_folder)
    print("Getting testing embeddings")
    test_image_features,  test_images  = get_image_features(test_image_ids, data_folder)



    return dict(
        train_questions          = np.array(train_questions),
        test_questions           = np.array(test_questions),
        train_image_features    = np.array(train_image_features),
        test_image_features     = np.array(test_image_features),
        train_images            = np.array(train_images),
        test_images             = np.array(test_images),
        train_answers           = np.array(train_answers),
        test_answers            = np.array(test_answers),
        word2idx                = word2idx,
        idx2word                = {v:k for k,v in word2idx.items()},
    )


def create_pickle(data_folder):
    with open(f'{data_folder}/data.p', 'wb') as pickle_file:
        pickle.dump(load_data(data_folder), pickle_file)
    print(f'Data has been dumped into {data_folder}/data.p!')


if __name__ == '__main__':
    ## Download this and put the Images and captions.txt indo your ../data directory
    ## Flickr 8k Dataset: https://www.kaggle.com/datasets/adityajn105/flickr8k?resource=download
    ## TODO: download the VQA dataset 
    data_folder = '../data'
    create_pickle(data_folder)