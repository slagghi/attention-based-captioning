# This code processes the images and the captions of the RSICD dataset
# mimicking the attention_tf_example file, which works on COCO dataset

# You'll generate plots of attention in order to see which parts of an image
# our model focuses on during captioning
import matplotlib.pyplot as plt

# Scikit-learn includes many helpful utilities
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import re
import numpy as np
import os
import time
import json
from glob import glob
from PIL import Image
import pickle
import copy

import tensorflow as tf
import random
from helpers import print_progress

# RSICD

print('Loading dataset files...')

RSICD_datapath='RSICD_images/'   

train_captions_file='dataset/captions_train.json'
val_captions_file='dataset/captions_val.json'
test_captions_file='dataset/captions_test.json'

train_filenames_file='dataset/filenames_train.json'
val_filenames_file='dataset/filenames_val.json'
test_filenames_file='dataset/filenames_test.json'

with open(train_captions_file,'r') as f:
    captions_train=json.load(f)
    
with open(val_captions_file,'r') as f:
    captions_val=json.load(f)
    
with open(test_captions_file,'r') as f:
    captions_test=json.load(f)

with open(train_filenames_file,'r') as f:
    filenames_train=json.load(f)

with open(val_filenames_file,'r') as f:
    filenames_val=json.load(f)
    
with open(test_filenames_file,'r') as f:
    filenames_test=json.load(f)

# CREATE A DATASET FOR TRAIN + VAL IMAGES


captions_trainval=captions_train+captions_val
filenames_trainval=filenames_train+filenames_val


all_captions = []
all_img_name_vector = []

print('Getting annotations...')
idxList=list()
idList=list()
for idx in range(len(captions_trainval)):
    annot=captions_trainval[idx]
    image_id=filenames_trainval[idx]
#for annot in captions_trainval:
#    idx=captions_trainval.index(annot)
#    image_id=filenames_trainval[idx]
    
    idxList.append(idx)
    idList.append(image_id)
    
    for cap in annot:
        caption='<start> ' + cap + ' <end>'
        full_image_path=os.path.join(RSICD_datapath,image_id)
        
        all_img_name_vector.append(copy.copy(full_image_path))
        all_captions.append(copy.copy(caption))

# Shuffle captions and image_names together
# Set a random state
train_captions, img_name_vector= shuffle(all_captions,
                                          all_img_name_vector,
                                          random_state=1)
    
def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path


def process_images():
    print('Loading image model...')

    image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                weights='imagenet')
    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output

    image_features_extract_model = tf.keras.Model(new_input, hidden_layer)


    # Get unique images
    encode_train = sorted(set(img_name_vector))


    # Feel free to change batch_size according to your system configuration
    batch_size=8
    
    print('Creating dataset...')

    image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
    image_dataset = image_dataset.map(
    load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_size)

    ctr=0
    print('Computing features...')
    for img, path in image_dataset:
        batch_features = image_features_extract_model(img)
        batch_features = tf.reshape(batch_features,
                              (batch_features.shape[0], -1, batch_features.shape[3]))
        ctr+=1
        print_progress(ctr*batch_size,len(filenames_trainval))
        for bf, p in zip(batch_features, path):
            path_of_feature = p.numpy().decode("utf-8")
            path_of_feature='processed_images/'+path_of_feature
            np.save(path_of_feature, bf.numpy())

def calc_max_length(tensor):
    return max(len(t) for t in tensor)

    
print('Tokenizing...')
# Choose the top K words from the vocabulary
top_k = 2000
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,
                                                  oov_token="<unk>",
                                                  filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
tokenizer.fit_on_texts(train_captions)
train_seqs = tokenizer.texts_to_sequences(train_captions)
    
tokenizer.word_index['<pad>'] = 0
tokenizer.index_word[0] = '<pad>'
    
# Create the tokenized vectors
train_seqs = tokenizer.texts_to_sequences(train_captions)
    
# Pad each vector to the max_length of the captions
# If you do not provide a max_length value, pad_sequences calculates it automatically
cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')
    
# Calculates the max_length, which is used to store the attention weights
max_length = calc_max_length(train_seqs)

print('Splitting the dataset')
# Create training and validation sets
img_name_train, img_name_val, cap_train, cap_val = train_test_split(img_name_vector,
                                                                        cap_vector,
                                                                        test_size=0.12,
                                                                        random_state=0)
    
print('Images in training set:')
print(len(img_name_train))
print('Captions in training set')
print(len(cap_train))
print('Images in validation set:')
print(len(img_name_val))
print('Captions in validations set:')
print(len(cap_val))
    
    
#print('Creating dataset for training...')
    
