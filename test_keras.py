from json import *
import os
from keras import backend as K
from keras.models import Model
from keras.layers import *
import keras
import csv
import numpy as np
from nltk.tokenize import word_tokenize
import itertools
import random



MAX_SENT_LENGTH = 30
MAX_SENTS = 50
len_word_dict = 10000
title_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')
embedding_layer = Embedding(len_word_dict, 300, trainable=True)
embedded_sequences = embedding_layer(title_input)

test_title_input = np.array([random.randint(0,100) for _ in range(30)])

a = embedding_layer(test_title_input)
print(a)