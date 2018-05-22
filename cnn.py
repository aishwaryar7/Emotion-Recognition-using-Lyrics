import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
np.random.seed(1337)
import matplotlib
matplotlib.use('Agg')

import keras
import pandas
import gensim
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, Flatten, Dropout
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model

# ------------------------------#
#  HYPERPARAMETERS TO CONTROL!
# ------------------------------#
MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 10000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.2
NUM_EPOCHS = 4
max_features = 40000
batch_size = 128

# prepare text samples and their labels
labels_index = {'happy': 0, 'angry': 1, 'sad': 2, 'relaxed': 3}
dataset = pandas.read_excel('ml_lyrics.xlsx')

# ----------------------------------------------------------------  #
#                       WORD2VEC MODEL                              #
# ----------------------------------------------------------------  #

word2vec_path = "GoogleNews-vectors-negative300.bin.gz"
word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)


def get_average_word2vec(tokens_list, vector, generate_missing=False, k=300):

    if len(tokens_list)<1:
        return np.zeros(k)
    if generate_missing:
        vectorized = [vector[word] if word in vector else np.random.rand(k) for word in tokens_list]
    else:
        vectorized = [vector[word] if word in vector else np.zeros(k) for word in tokens_list]
    length = len(vectorized)
    summed = np.sum(vectorized, axis=0)
    averaged = np.divide(summed, length)
    return averaged


def get_word2vec_embeddings(vectors, dataset, generate_missing=False):

    embeddings = dataset['Lyrics'].apply(lambda x: get_average_word2vec(x, vectors, generate_missing=generate_missing))
    return list(embeddings)


training_embeddings = get_word2vec_embeddings(word2vec, dataset, generate_missing=True)

# ----------------------------------------------------------------  #
#                       PREPARING THE DATA                          #
# ----------------------------------------------------------------  #

texts = dataset["Lyrics"].tolist()
labels = dataset["Mood"].tolist()

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

for i in range(len(labels)):
    labels[i] = labels_index[labels[i]]

labels = keras.utils.to_categorical(labels, num_classes=4)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]

# ---------------------------------------------------------#
#                     WORD EMBEDDINGS                      #
# ---------------------------------------------------------#

embeddings_index = {}

nb_words = min(MAX_NB_WORDS, len(word_index))
embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    if i > MAX_NB_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector


# ----------------------------------------------------- #
#                 CONVOLUTIONAL NEURAL NET              #
# ----------------------------------------------------- #


def ConvNet(embeddings, max_sequence_length, num_words, embedding_dim, labels_index, trainable=False):
    embedding_layer = Embedding(num_words + 1,
                                embedding_dim,
                                weights=[embeddings],
                                input_length=max_sequence_length,
                                trainable=trainable)

    sequence_input = Input(shape=(max_sequence_length,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)

    convs = []
    filter_sizes = [3, 4, 5]

    for filter_size in filter_sizes:
        l_conv = Conv1D(filters=128, kernel_size=filter_size, activation='relu')(embedded_sequences)
        l_pool = MaxPooling1D(pool_size=3)(l_conv)
        convs.append(l_pool)

    conv = Conv1D(filters=128, kernel_size=3, activation='relu')(embedded_sequences)
    pool = MaxPooling1D(pool_size=3)(conv)

    x = Dropout(0.5)(pool)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)

    preds = Dense(labels_index, activation='sigmoid')(x)

    model = Model(sequence_input, preds)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
    #model.summary()

    return model

# ------------------------------------------------------ #
#                    TRAINING THE NETWORK                #
# ------------------------------------------------------ #

def plot_training(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    epochs = range(len(acc))

    plt.figure()

    plt.plot(epochs, acc, 'r.')
    plt.plot(epochs, val_acc, 'r')
    plt.title('Training and validation accuracy')

    plt.savefig('CNN_Plot.png')


model = ConvNet(embedding_matrix, MAX_SEQUENCE_LENGTH, nb_words, EMBEDDING_DIM, len(list(labels_index)), False)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    iteration = 1
    for i in range(iteration):
        print("Iteration : ", i)
        s = np.arange(x_train.shape[0])
        np.random.shuffle(s)
        xTr = x_train[s]
        yTr = y_train[s]

        history = model.fit(xTr, yTr, validation_data=(x_val, y_val), epochs=10, batch_size=128)

        #print(history.history.keys())

        score, acc = model.evaluate(x_val, y_val,  batch_size=866)

        print('Test score for CNN model:', score)
        print('Test accuracy for CNN model:', acc)

        plot_training(history)

