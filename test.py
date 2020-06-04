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
class Attention(Layer):
    
    def __init__(self, nb_head, size_per_head, **kwargs):
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.output_dim = nb_head*size_per_head
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.WQ = self.add_weight(name='WQ',
                                  shape=(input_shape[0][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WK = self.add_weight(name='WK',
                                  shape=(input_shape[1][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WV = self.add_weight(name='WV',
                                  shape=(input_shape[2][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        super(Attention, self).build(input_shape)

    def Mask(self, inputs, seq_len, mode='mul'):
        if seq_len == None:
            return inputs
        else:
            mask = K.one_hot(seq_len[:, 0], K.shape(inputs)[1])
            mask = 1 - K.cumsum(mask, 1)
            for _ in range(len(inputs.shape)-2):
                mask = K.expand_dims(mask, 2)
            if mode == 'mul':
                return inputs * mask
            if mode == 'add':
                return inputs - (1 - mask) * 1e12

    def call(self, x):
        if len(x) == 3:
            Q_seq, K_seq, V_seq = x
            Q_len, V_len = None, None
        elif len(x) == 5:
            Q_seq, K_seq, V_seq, Q_len, V_len = x
        Q_seq = K.dot(Q_seq, self.WQ)
        Q_seq = K.reshape(
            Q_seq, (-1, K.shape(Q_seq)[1], self.nb_head, self.size_per_head))
        Q_seq = K.permute_dimensions(Q_seq, (0, 2, 1, 3))
        K_seq = K.dot(K_seq, self.WK)
        K_seq = K.reshape(
            K_seq, (-1, K.shape(K_seq)[1], self.nb_head, self.size_per_head))
        K_seq = K.permute_dimensions(K_seq, (0, 2, 1, 3))
        V_seq = K.dot(V_seq, self.WV)
        V_seq = K.reshape(
            V_seq, (-1, K.shape(V_seq)[1], self.nb_head, self.size_per_head))
        V_seq = K.permute_dimensions(V_seq, (0, 2, 1, 3))
        A = K.batch_dot(Q_seq, K_seq, axes=[3, 3]) / self.size_per_head**0.5
        A = K.permute_dimensions(A, (0, 3, 2, 1))
        A = self.Mask(A, V_len, 'add')
        A = K.permute_dimensions(A, (0, 3, 2, 1))
        A = K.softmax(A)
        O_seq = K.batch_dot(A, V_seq, axes=[3, 2])
        O_seq = K.permute_dimensions(O_seq, (0, 2, 1, 3))
        O_seq = K.reshape(O_seq, (-1, K.shape(O_seq)[1], self.output_dim))
        O_seq = self.Mask(O_seq, Q_len, 'mul')
        return O_seq

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.output_dim)


MAX_SENT_LENGTH = 30
MAX_SENTS = 50
len_word_dict = 10000
title_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')
embedding_layer = Embedding(len_word_dict, 300, trainable=True)
embedded_sequences = embedding_layer(title_input)
d_emb = Dropout(0.2)(embedded_sequences)
selfatt = Attention(20, 20)([d_emb, d_emb, d_emb])
selfatt = Dropout(0.2)(selfatt)
attention = Dense(200, activation='tanh')(selfatt)
attention = Flatten()(Dense(1)(attention))
attention_weight = Activation('softmax')(attention)
rep = Dot((1, 1))([selfatt, attention_weight])
titleEncoder = Model([title_input], rep)

news_input = Input((MAX_SENTS, MAX_SENT_LENGTH,))
news_encoders = TimeDistributed(titleEncoder)(news_input)
news_encoders = Dropout(0.2)(Attention(20, 20)(
    [news_encoders, news_encoders, news_encoders]))
candidates = keras.Input((1+npratio, MAX_SENT_LENGTH,))
candidate_vecs = TimeDistributed(titleEncoder)(candidates)
news_attention = Dense(200, activation='tanh')(news_encoders)
news_attention = Flatten()(Dense(1)(news_attention))
news_attention_weight = Activation('softmax')(news_attention)
userrep = Dot((1, 1))([news_encoders, news_attention_weight])
logits = dot([userrep, candidate_vecs], axes=-1)
logits = Activation(keras.activations.softmax)(logits)
model = Model([candidates, news_input], logits)
model.compile(loss=['categorical_crossentropy'],
              optimizer='adam', metrics=['acc'])

candidate_one = keras.Input((MAX_SENT_LENGTH,))
candidate_one_vec = titleEncoder([candidate_one])
score = Activation(keras.activations.sigmoid)(
    dot([userrep, candidate_one_vec], axes=-1))
modeltest = keras.Model([candidate_one, news_input], score)