import pandas as pd
import numpy as np
import nltk
import re
import string

raw_hate = pd.read_csv('./labeled_data.csv')

raw_hate = raw_hate[['class', 'tweet']]

labels = raw_hate['class']

tweets = raw_hate['tweet']

tweets = tweets.as_matrix(columns=None)

def no_punc(tweets_arr):
    translator = str.maketrans('', '', string.punctuation)
    
    tweets_no_punc = [t.translate(translator) for t in tweets_arr]
    
    return tweets_no_punc

tweets = no_punc(tweets)

tweets = [t.lower() for t in tweets]

stopwords=stopwords = nltk.corpus.stopwords.words("english")

other_stops = ["ff", "rt", "amp"]
stopwords.extend(other_stops)

def remove_stop_words_from_tweet(a_tweet_word_array):
    stopwords_free_tweet = [word for word in a_tweet_word_array if word not in stopwords]
    return stopwords_free_tweet

def remove_stop_words(tweets_list):
    words_in_tweets = [t.split() for t in tweets_list]
    stopwords_removed = [remove_stop_words_from_tweet(tweet) for tweet in words_in_tweets]
    stopword_free_tweets = [' '.join(split_tweet) for split_tweet in stopwords_removed]
    return stopword_free_tweets

tweets = remove_stop_words(tweets)

df_for_gensim = pd.DataFrame({'labels':labels, 'tweets':tweets})

df_for_gensim['tweets'] = df_for_gensim['tweets'].str.replace('\d+', '')

np.savetxt('./text_for_gensim.txt', df_for_gensim.values,
           fmt=['%.0f', '%s'],
           delimiter='\t')
