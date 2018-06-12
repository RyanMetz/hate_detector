# hate_detector

An implementation of a neural network topology for detecting hate speech. It averages around 90% categorical accuracy with test sets.

This project uses a classic NLP techniques and a neural network to classify a tweet as either: 0 - hate speech 1 - offensive language 2 - neither.

There are 2 scripts with slightly different implementations of the neural network. Both scripts require Python 3.6, Numpy, Pandas, NLTK, gensim (with the Google word2vec binary), TensorFLow, and Keras. The first, hate_speech_detector_NN.py requires a GPU that supports CUDA and the full CUDA environment installed, including CuDNN. If you don't have such a setup, a second script, hate_detector_NO_CUDA.py will run on a CPU without CUDA, using a normal Keras GRU layer. The only real difference for the purpose of running the script is that the regular GRU is much slower.

You will need the google news word2vec binary file, which can be downloaded here: https://drive.google.com/uc?export=download&confirm=nJWr&id=0B7XkCwpI5KDYNlNUTTlSS21pQmM Unzip that file and put the binary in the directory with the python script. I considered having the file downloaded and unzipped in the script but decided that doing that with a 1.5 gig file would be poor form.

The network is an implementation of a neural network topology published in a conference paper: Zhang, Ziqi & Robinson, D & Tepper, Jonathan. (2018). Detecting hate speech on Twitter using a convolution-GRU based deep neural network. 

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

A csv file - "labeled_data.csv", contains the data. It is preprocessed by removing unnecessary columns, punctuation, numbers, and the English stopwords from the NLTK, along with "rt", "ff", and "amp". After making the tweets lowcased they are tokenized with the Keras tokenizer. Only the most 10,000 common words were kept, which removed not only uncommon words, but also user names. Finally, they are transformed into 300 dimension word vectors with word2vec using the Google News corpus. These vectors are passed to the network. An initial 80:20 train test split is performed. The test set is then divided in half into a validation set and a withheld set for scoring after training was complete. After training, the topology and weights are saved to file, hate_detector_trained.h5.
