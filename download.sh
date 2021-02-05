#!/usr/bin/env bash

mkdir embeddings
cd embeddings

# download glove embeddings
wget http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip glove.840B.300d.zip

# download counter-fitted embeddings
wget https://github.com/nmrksic/counter-fitting/raw/master/word_vectors/counter-fitted-vectors.txt.zip
unzip counter-fitted-vectors.txt.zip

cd ..

