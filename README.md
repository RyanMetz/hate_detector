# hate_detector
An implementation of a neural network topology for detecting hate speech

This project uses a classic NLP techniques and a neural network to classify a tweet as either: 0 - hate speech 1 - offensive language 2 - neither.

It is an implementation of a neural network topology published in a conference paper: Zhang, Ziqi & Robinson, D & Tepper, Jonathan. (2018). Detecting hate speech on Twitter using a convolution-GRU based deep neural network. 

It makes use of data from:
@inproceedings{hateoffensive,
title = {Automated Hate Speech Detection and the Problem of Offensive Language},
author = {Davidson, Thomas and Warmsley, Dana and Macy, Michael and Weber, Ingmar}, 
booktitle = {Proceedings of the 11th International AAAI Conference on Web and Social Media},
series = {ICWSM '17},
year = {2017},
location = {Montreal, Canada},
pages = {512-515}
}

A csv file containing the tweets, labeled as above, is preprocessed by removing punctuation, numbers, and the English stopwords from the NLTK, along with "rt", "ff", and "amp".
After making the tweets lowcased they were then stemmed with the English Snowball Stemmer, and tokenized with the Keras tokenizer.
Words that appeared less than 5 times were disregarded.
The tweets where then fed to gensim and turned into vectors by a model trained on the Google News corpus. 
