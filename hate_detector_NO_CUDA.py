import os
import collections
import pandas as pd
import numpy as np
import logging
import nltk

import gensim
from gensim.models import KeyedVectors

from keras import backend
from keras.utils import np_utils
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Embedding, Dropout, Conv1D, MaxPooling1D, GRU, GlobalMaxPooling1D, Dense
from keras import regularizers
from keras import metrics
import h5py


INPUT_FILE = './text_for_gensim.txt'

WORD2VEC_MODEL = "./GoogleNews-vectors-negative300.bin"
VOCAB_SIZE = 10000
EMBEDDING_DIM = 300
NUM_FILTERS = 100
NUM_WORDS = 4
BATCH_SIZE = 300
NUM_EPOCHS = 3

counter = collections.Counter()
fin = open(INPUT_FILE, 'r')
maxlen = 0
for line in fin:
    _, sent = line.strip().split('\t')
    words = [x.lower() for x in nltk.word_tokenize(sent)]
    if len(words) > maxlen:
        maxlen = len(words)
    for word in words:
        counter[word] += 1
fin.close()

word2index = collections.defaultdict(int)
for wid, word in enumerate(counter.most_common(VOCAB_SIZE)):
    word2index[word[0]] = wid + 1
vocab_size = len(word2index) + 1
index2word = {v:k for k, v in word2index.items()}

xs, ys = [], []
fin = open(INPUT_FILE, 'r')
for line in fin:
    label, sent = line.strip().split('\t')
    ys.append(int(label))
    words = [x for x in nltk.word_tokenize(sent)]
    wids = [word2index[word] for word in words]
    xs.append(wids)
fin.close()
X = pad_sequences(xs, maxlen=100)
y = np_utils.to_categorical(ys)

FIRST_VALIDATION_SPLIT = 0.2
SECOND_VALIDATION_SPLIT = 0.5
MAX_SEQUENCE_LENGTH = 100

first_indices = np.arange(X.shape[0])
np.random.shuffle(first_indices)
data = X[first_indices]
labels = y[first_indices]
nb_validation_samples_1 = int(FIRST_VALIDATION_SPLIT * data.shape[0])

X_train = data[:-nb_validation_samples_1]
y_train = labels[:-nb_validation_samples_1]
X_val_to_divide = data[-nb_validation_samples_1:]
y_val_to_divide = labels[-nb_validation_samples_1:]

second_indices = np.arange(X_val_to_divide.shape[0])
np.random.shuffle(second_indices)
X_val_to_divide = X_val_to_divide[second_indices]
y_val_to_divide = y_val_to_divide[second_indices]
nb_validation_samples_2 = int(SECOND_VALIDATION_SPLIT * X_val_to_divide.shape[0])

X_val = X_val_to_divide[:-nb_validation_samples_2]
y_val = y_val_to_divide[:-nb_validation_samples_2]
X_hold = X_val_to_divide[-nb_validation_samples_2:]
y_hold = y_val_to_divide[-nb_validation_samples_2:]

labels = to_categorical(np.asarray(y))

word2vec = gensim.models.KeyedVectors.load_word2vec_format(WORD2VEC_MODEL, 
                                                           binary=True)

embedding_weights = np.zeros((vocab_size, EMBEDDING_DIM))
for word, index in word2index.items():
    try:
        embedding_weights[index, :] = word2vec[word]
    except KeyError:
        pass

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

BATCH = 300

model = Sequential()

model.add(Embedding(vocab_size, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH, weights=[embedding_weights]))

model.add(Dropout(0.2))

model.add(Conv1D(filters=100, kernel_size=4, padding='same', activation='relu'))

model.add(MaxPooling1D(pool_size=4))

model.add(GRU(100, return_sequences=True))

model.add(GlobalMaxPooling1D(input_shape=(25, 100)))

model.add(Dense(3, activation='softmax', activity_regularizer=regularizers.l1_l2(0.01)))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

print(model.summary())

history = model.fit(X_train, y_train,
                    batch_size=BATCH,
                    epochs=10,
                    validation_data=(X_val, y_val))

model.save_weights('./hate_detector_trained.h5')

model.evaluate(x=X_hold, y=y_hold, batch_size=300, verbose=0)

