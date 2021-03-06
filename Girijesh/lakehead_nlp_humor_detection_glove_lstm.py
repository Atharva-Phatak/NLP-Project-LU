# -*- coding: utf-8 -*-
"""Lakehead_NLP_Humor_Detection_GloVe_LSTM.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1TvBZKQ1owZe69yHW3vbWZI229GJmiB97
"""

#Mount drive to access files in gdrive
from google.colab import drive
drive.mount('/content/gdrive', force_remount=True)

# Import libraries
import pandas as pd
import numpy as np
import string
import re
import keras
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelBinarizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation, Input
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras import backend as K
from keras.models import Model

# reading semval datasets
df_train = pd.read_csv("/content/gdrive/MyDrive/Lakehead_NLP_Project/train.csv")
df_test = pd.read_csv("/content/gdrive/MyDrive/Lakehead_NLP_Project/dev.csv")
df_gold = pd.read_csv("/content/gdrive/MyDrive/Lakehead_NLP_Project/gold-test-27446.csv")

# Commented out IPython magic to ensure Python compatibility.
# # text preprocessing
# %%time
# def text_preprocessing(text):
#   text = text.lower()
#   text = re.sub(r"http\S+", " ", text)
#   text = re.sub(' +', ' ', text)
#   
#   return text

#applying text preprocessing on all dataset

# train data
df_train_1 = df_train.copy()
df_train_1['text'] = df_train_1['text'].apply(text_preprocessing)

#test data
df_test_1 = df_test.copy()
df_test_1['text'] = df_test_1['text'].apply(text_preprocessing)

#gold data
df_gold_1 = df_gold.copy()
df_gold_1['text'] = df_gold_1['text'].apply(text_preprocessing)

# intializing tokenizer and fitting on train, test and dev data and padding to each sequences

vocabulary_size = 20000
maxlen = 200

tokenizer = Tokenizer(num_words= vocabulary_size)

#creating vocabulary on train data
tokenizer.fit_on_texts(df_train_1['text'])

#generating sequence of tokens on train, text, gold data
train_sequences = tokenizer.texts_to_sequences(df_train_1['text'])
test_sequences = tokenizer.texts_to_sequences(df_test_1['text'])
gold_sequences = tokenizer.texts_to_sequences(df_gold_1['text'])

#padding the sequences
train_pad = pad_sequences(train_sequences, maxlen=maxlen)
test_pad = pad_sequences(test_sequences, maxlen=maxlen)
gold_pad = pad_sequences(gold_sequences, maxlen=maxlen)

# reading glove embedding and creating weight matrix 
def read_glove_embedding(vocabulary_size, df, glove_path, embedding_size, tokenizer):

  embeddings_index = dict()

  f = open(glove_path)
  for line in f:
      values = line.split()
      word = values[0]
      coefs = np.asarray(values[1:], dtype='float32')
      embeddings_index[word] = coefs
  f.close()
  print('Loaded %s word vectors.' % len(embeddings_index))

  embedding_matrix = np.zeros((vocabulary_size, embedding_size))
  for word, index in tokenizer.word_index.items():
      if index > vocabulary_size - 1:
          break
      else:
          embedding_vector = embeddings_index.get(word)
          if embedding_vector is not None:
              embedding_matrix[index] = embedding_vector

  return embedding_matrix

#create embedding for train data
vocabulary_size = 20000
df_new = df_train_1
glove_path = '/content/gdrive/MyDrive/Lakehead_NLP_Project/glove.6B.300d.txt'
embedding_size = 300

embedding_matrix = read_glove_embedding(vocabulary_size, df_new, glove_path,  embedding_size, tokenizer)

# defining metrices to evaluate the model
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

# constructing model

embedding_input_shape = 300
model_glove = Sequential()
model_glove.add(Embedding(vocabulary_size, embedding_input_shape, input_length=150, weights=[embedding_matrix], trainable=False))
model_glove.add(Dropout(0.2))
model_glove.add(Conv1D(64, 5, activation='relu'))
model_glove.add(MaxPooling1D(pool_size=4))
model_glove.add(LSTM(300))
model_glove.add(Dense(1, activation='sigmoid'))
model_glove.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy',f1_m, precision_m, recall_m])

# Commented out IPython magic to ensure Python compatibility.
# %%time
# #fitting model on train data
# model_glove.fit(train_pad, np.array(df_train_1['is_humor']), validation_data=(test_pad, np.asarray(df_test_1['is_humor'])), epochs = 10, batch_size=128)

# evaluating model on gold data
gold_lstm_results = model_glove.evaluate(gold_pad, np.asarray(df_gold_1['is_humor']), verbose=0, batch_size=128)

print(f'Gold Data accuracy: {gold_lstm_results[1]*100:0.2f}')
print(f'Gold Data F1 Score: {gold_lstm_results[2]}')

