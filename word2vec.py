import os
import time
import plotly
import numpy as np
import pandas as pd
import multiprocessing
import sklearn.manifold
import plotly.graph_objs as go
import gensim.models.word2vec as w2v
from plotly.graph_objs import Layout
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

dataset = pd.read_excel("ml_lyrics.xlsx", header=0)

# Split the song into words and add each song as an array
corpus = []
count = 0
num_words = 0

# dimension for the word2vec model
num_features = 50
min_word_count = 1
max_word_count = 1000
num_parallel_threads = multiprocessing.cpu_count()
context_size = 7
sample = 1e-1
seed = 1

for i in dataset['Lyrics']:
    split_words = i.lower().split()
    num_words = len(split_words) + num_words
    corpus.append(split_words)


lyrics2vec = w2v.Word2Vec(
    sg=1,
    seed=seed,
    workers=num_parallel_threads,
    size=num_features,
    min_count=min_word_count,
    window=context_size,
    sample=sample
)

lyrics2vec.build_vocab(corpus)
print("Length of text corpus : %s " % len(corpus))
print("Number of words in the text corpus : %s " % num_words)


##################################################################################
# Train the lyrics2vec model

##################################################################################

start_time = time.time()
lyrics2vec.train(corpus, total_examples=len(corpus), epochs=1)

if not os.path.exists("trained"):
    os.makedirs("trained")

lyrics2vec.save(os.path.join("trained", "lyrics2vectors.w2v"))

print("Time taken to train the model is %s seconds " % (time.time() - start_time))

lyrics2vec = w2v.Word2Vec.load(os.path.join("trained", "lyrics2vectors.w2v"))

##################################################################################
# Normalized vector sum for each song

#################################################################################

def songvector(row):

    vector_sum = 0
    words = row.lower().split()
    for word in words:
        vector_sum = vector_sum + lyrics2vec[word]
    vector_sum = vector_sum.reshape(1, -1)
    normalised_vector_sum = sklearn.preprocessing.normalize(vector_sum)
    return normalised_vector_sum

dataset['song_vector'] = dataset['Lyrics'].apply(songvector)

song_vectors = []
train, test = train_test_split(dataset, test_size=0.0)

for song_vector in train['song_vector']:
    song_vectors.append(song_vector)

X = np.array(song_vectors).reshape((4330, 50))

######################################################################################
# T-SNE

######################################################################################

tsne = sklearn.manifold.TSNE(n_components=2, n_iter=300, random_state=0, verbose=2)
all_word_vectors_matrix_2d = tsne.fit_transform(X)
df = pd.DataFrame(all_word_vectors_matrix_2d, columns=['X', 'Y'])
df.head(10)

train.head()

df.reset_index(drop=True, inplace=True)
train.reset_index(drop=True, inplace=True)

# Join the dataframe to get the X, Y co-ordinate
songs_2d = pd.concat([train, df], axis=1)
songs_2d.head()

# Plotting the result on a scatter plot
trace1 = go.Scatter(
    y=songs_2d['Y'],
    x=songs_2d['X'],
    text=songs_2d['Title'],
    mode='markers',
    marker=dict(
        size='10',
        color=np.random.randn(5000),
        colorscale='Electric',
        showscale=True,
    )
)

image = plotly.offline.plot({
    "data": [trace1],
    "layout": Layout(title="Scatter Plot")},
    auto_open=True, image='png', image_filename='Scatter_Plot',
    output_type='file', image_width=800, image_height=600,
    filename='Scatter Plot.html', validate=False
)


