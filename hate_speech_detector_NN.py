import os
import collections
import pandas as pd
import numpy as np
import logging
import gensim
from gensim.models import KeyedVectors

from keras import backend
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical, plot_model
from keras.models import Sequential
from keras.layers import Embedding, Dropout, Conv1D, MaxPooling1D, CuDNNGRU, GlobalMaxPooling1D, Dense
from keras import regularizers
from keras import metrics
from keras.utils.vis_utils import model_to_dot
import h5py

from IPython.display import SVG

formatted_tweets_df = pd.read_csv('./df_for_gensim.csv')
hate_tweets = pd.read_csv('./raw_hate_tweets_class.csv')

X = formatted_tweets_df
y = hate_tweets['class']

X = X.values.tolist()
y = y.values.tolist()

X = [item for sublist in X for item in sublist]

# The below is in here to give the NN something. Once word vectors are up and running I will not be tokenizing anything.
tokenizer = Tokenizer(num_words=3500000)
tokenizer.fit_on_texts(X)
sequences = tokenizer.texts_to_sequences(X)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

FIRST_VALIDATION_SPLIT = 0.2
SECOND_VALIDATION_SPLIT = 0.5
MAX_SEQUENCE_LENGTH = 100
VOCAB_SIZE = 31823
EMBEDDING_DIM = 300

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

np.max(data)

labels = to_categorical(np.asarray(y))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

first_indices = np.arange(data.shape[0]) # Once word vectors are made the split will happen after the words have all been transformed into vects.
np.random.shuffle(first_indices)
data = data[first_indices]
labels = labels[first_indices]
nb_validation_samples_1 = int(FIRST_VALIDATION_SPLIT * data.shape[0])

X_train = data[:-nb_validation_samples_1]
y_train = labels[:-nb_validation_samples_1]
X_val_to_divide = data[-nb_validation_samples_1:]
y_val_to_divide = labels[-nb_validation_samples_1:]

X_val_to_divide.shape
y_val_to_divide.shape

second_indices = np.arange(X_val_to_divide.shape[0])
np.random.shuffle(second_indices)
X_val_to_divide = X_val_to_divide[second_indices]
y_val_to_divide = y_val_to_divide[second_indices]
nb_validation_samples_2 = int(SECOND_VALIDATION_SPLIT * X_val_to_divide.shape[0])

X_val = X_val_to_divide[:-nb_validation_samples_2]
y_val = y_val_to_divide[:-nb_validation_samples_2]
X_hold = X_val_to_divide[-nb_validation_samples_2:]
y_hold = y_val_to_divide[-nb_validation_samples_2:]

counter = collections.Counter()

word2index = collections.defaultdict(int)
for wid, word in enumerate(counter.most_common(VOCAB_SIZE)):
    word2index[word[0]] = wid + 1
vocab_sz = len(word2index) + 1
index2word = {v:k for k, v in word2index.items()}

# The word2vec model is put together, it's just a matter of using it on my corpus to create the WVs.
word2vec = gensim.models.KeyedVectors.load_word2vec_format('./goog_vec/GoogleNews-vectors-negative300.bin', binary=True)
embedding_weights = np.zeros((vocab_sz, EMBEDDING_DIM))
for word, index in word2index.items():
    try:
        embedding_weights[index, :] = word2vec[word]
    except KeyError:
        pass

print(VOCAB_SIZE)
print(EMBEDDING_DIM)
print(embedding_weights)
print(len(word_index) + 1)


# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

print(embedding_weights.shape)


# This is a slightly modified implementation of a paper published at a conference in March 2018
# Title: Detecting Hate Speech on Twitter Using aConvolution-GRU Based Deep Neural Network
# Authors: Ziqi Zhang, David Robinson, and Jonathan Tepper
BATCH = 300

model = Sequential()

model.add(Embedding(VOCAB_SIZE + 1, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH, trainable=False))

model.add(Dropout(0.2))

model.add(Conv1D(filters=100, kernel_size=4, padding='same', activation='relu'))

model.add(MaxPooling1D(pool_size=4))

model.add(CuDNNGRU(100, return_sequences=True))

model.add(GlobalMaxPooling1D(input_shape=(25, 100)))

model.add(Dense(3, activation='softmax', activity_regularizer=regularizers.l1_l2(0.01)))


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])


print(model.summary())


SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))


history = model.fit(X_train, y_train,
          batch_size=BATCH,
          epochs=10,
          validation_data=(X_val, y_val))

model.save_weights('./trained_model.h5')

train_loss = history.history['loss']
test_loss = history.history['val_loss']

plt.ylabel('loss')
plt.xlabel('epoch')
plt.plot(train_loss, label='Train Loss')
plt.plot(test_loss, label='Test Loss')
plt.legend();

preds = model.predict_classes(X_hold)

# Once the WVs are being used and the model is truly complete I will use an F1 score for multi-class classification to measure model quality.
