#!/usr/bin/env python
# coding: utf-8

# In[3]:

import numpy as np
from numpy import array
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


import seaborn as sns
sns.set()
from collections import Counter
from nltk.corpus import stopwords

import keras
from sklearn.model_selection import train_test_split
import string
import os
import time
from PIL import Image
import glob
from pickle import dump, load
from time import time
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector,Activation, Flatten, Reshape, concatenate, Dropout, BatchNormalization
from keras.optimizers import Adam, RMSprop

from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras import Input, layers
from keras import optimizers
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import load_model, Model


# In[4]:


model = load_model("model.h5")



# In[7]:


import pickle
from pickle import dump,load
with open("vocab.pkl", "rb") as voc:
    vocab = pickle.load(voc)
with open("covab.pkl", "rb") as cov:
    covab = pickle.load(cov)

# In[8]:


# image_model = keras.applications.InceptionV3(include_top=False,weights='imagenet')
# new_input = image_model.input
# hidden_layer = image_model.layers[-1].output

# base_model = keras.Model(new_input, hidden_layer)
# base_model

# In[9]:


def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = preprocess_input(img)
    return img, image_path


# In[11]:


BATCH_SIZE = 32
BUFFER_SIZE = 1000
embedding_dim = 256
units = 512


# In[12]:


class Encoder(Model):
    def __init__(self,embed_dim):
        super(Encoder, self).__init__()
        self.dense = tf.keras.layers.Dense(embed_dim) #build your Dense layer with relu activation
        
    def call(self, features):
        features =  self.dense(features) # extract the features from the image shape: (batch, 8*8, embed_dim)
        features =  tf.keras.activations.relu(features, alpha=0.01, max_value=None, threshold=0)
        return features

encoder = Encoder(embedding_dim)


# In[13]:


class Attention_model(Model):
    def __init__(self, units):
        super(Attention_model, self).__init__()
        self.W1 = tf.keras.layers.Dense(units) 
        self.W2 = tf.keras.layers.Dense(units) 
        self.V = tf.keras.layers.Dense(1) 
        self.units=units

    def call(self, features, hidden):
        hidden_with_time_axis = hidden[:, tf.newaxis]
        score = tf.keras.activations.tanh(self.W1(features) + self.W2(hidden_with_time_axis))  
        attention_weights = tf.keras.activations.softmax(self.V(score), axis=1) 
        context_vector = attention_weights * features 
        context_vector = tf.reduce_sum(context_vector, axis=1)  
        return context_vector, attention_weights


# In[14]:


class Decoder(Model):
    def __init__(self, embed_dim, units, vocab_size):
        super(Decoder, self).__init__()
        self.units=units
        self.attention = Attention_model(self.units) #iniitalise your Attention model with units
        self.embed = tf.keras.layers.Embedding(vocab_size, embed_dim) #build your Embedding layer
        self.gru = tf.keras.layers.GRU(self.units,return_sequences=True,return_state=True,recurrent_initializer='glorot_uniform')
        self.d1 = tf.keras.layers.Dense(self.units) #build your Dense layer
        self.d2 = tf.keras.layers.Dense(vocab_size) #build your Dense layer
        

    def call(self,x,features, hidden):
        context_vector, attention_weights = self.attention(features, hidden) #create your context vector & attention weights from attention model
        embed = self.embed(x) # embed your input to shape: (batch_size, 1, embedding_dim)
        embed = tf.concat([tf.expand_dims(context_vector, 1), embed], axis = -1) # Concatenate your input with the context vector from attention layer. Shape: (batch_size, 1, embedding_dim + embedding_dim)
        output,state = self.gru(embed) # Extract the output & hidden state from GRU layer. Output shape : (batch_size, max_length, hidden_size)
        output = self.d1(output)
        output = tf.reshape(output, (-1, output.shape[2])) # shape : (batch_size * max_length, hidden_size)
        output = self.d2(output) # shape : (batch_size * max_length, vocab_size)
        
        return output, state, attention_weights
    
    def init_state(self, batch_size):
        return tf.zeros((batch_size, self.units))


# In[24]:


vocab_size = 9862
decoder=Decoder(embedding_dim, units, vocab_size)


# In[16]:


attention=Attention_model(units)


# In[17]:

tokenizer = keras.preprocessing.text.Tokenizer(num_words=5000,oov_token="<unk>",filters='!"#$%&()*+.-/:;=?@[\]^_`{|}~ ')

def evaluate(image):
    attention_plot = np.zeros((40, 64))

    hidden = decoder.init_state(batch_size=1)

    temp_input = tf.expand_dims(load_image(image)[0], 0)
    img_tensor_val = model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

    features = encoder(img_tensor_val)

    dec_input = tf.expand_dims([vocab['<start>']], 0)
    result = []

    for i in range(40):
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)

        attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()

        predicted_id = tf.argmax(predictions[0]).numpy()
        result.append(covab[predicted_id])

        if covab[predicted_id] == '<end>':
            return result, attention_plot,predictions

        dec_input = tf.expand_dims([predicted_id], 0)

    attention_plot = attention_plot[:len(result), :]
    return result, attention_plot,predictions


# In[18]:


# def filt_text(text):
#     filt=['<start>','<unk>','<end>'] 
#     temp= text.split()
#     [temp.remove(j) for k in filt for j in temp if k==j]
#     text=' '.join(temp)
#     return text


# In[21]:


from gtts import gTTS
from playsound import playsound
from IPython import display


# In[19]:


# def pred_caption_audio(random, autoplay=False, weights=(0.5, 0.5, 0, 0)) :

    
    # test_image="C:/Users/debje/Desktop/Images/106490881_5a2dd9b7bd.jpg"
    

    # return test_image
    


# In[22]:


def caption_this_image(input_img): 

    
    
    result, attention_plot, pred_test = evaluate(input_img)
     
    pred_caption=' '.join(result).rsplit(' ', 1)[0]

    candidate = pred_caption.split()

    print ('Prediction Caption:', pred_caption)
#     plot_attmap(result, attention_plot, test_image)
    speech = gTTS('Predicted Caption : ' + pred_caption, lang = 'en', slow = False)
    speech.save('voice.mp3')
    audio_file = 'voice.mp3'

    display.display(display.Audio(audio_file, rate = None, autoplay = True))
    # keras.backend.clear_session()
    return pred_caption


# In[ ]:




