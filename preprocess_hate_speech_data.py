import IPython
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from word_processing_functions import *

get_ipython().run_line_magic('matplotlib', 'auto') # - Inlines text in a Jupyter Notebook / IPython

raw_hate = pd.read_csv('./labeled_data.csv')

print(f'First 5 rows of raw data: \n{raw_hate.head()}')

# "Unnamed" is just a redundant index; 
# "count" is the number of users who labeled a tweet; 
# "hate_speech" "offensive_language" and "neither", are the number of votes a tweet got for each of those classes;
# "class" is the final label assignment determined by majority vote;
# "tweet" is the text of each tweet. 

# In "class" the values represent:
# 0 - hate speech
# 1 - offensive  language
# 2 - neither

# Because I only care about the text and the class, I'm dropping the rest.
raw_hate = raw_hate[['class', 'tweet']]

# Check number of null elements in dataframe
print(f'Number of null elements in the dataframe: \n{raw_hate.isnull().sum()}') 

# Check number of rows and columns in the dataframe
print(f'Rows and columns in the dataframe: \n{raw_hate.shape}') 

# Check number of unique values in each column
print(f'Unique values in each column: \n{raw_hate.nunique()}') 

# raw_hate['class'].hist() - In Jupyter Notebook, prints an inlined histogram of the number of tweets in each class

# Prints exact number of tweets in each class
print(f'Number of tweets of each class: \n{raw_hate["class"].value_counts()}') 

# Save to CSV as backup
raw_hate.to_csv('./raw_hate_tweets_class.csv', index=False)

tweets = raw_hate['tweet']
tweets = tweets.as_matrix(columns=None)

print(f'First 5 unformatted tweets: \n{tweets[:5]}')

def format_tweets(tweets_list):
    tweets_all_lowercase = [t.lower() for t in tweets_list]
    tweets_no_punc = remove_punctuation(tweets_all_lowercase)
    tweets_no_numbers = remove_numbers(tweets_no_punc)

    words_in_tweets = [t.split() for t in tweets_no_numbers]
    stopwords_removed = [remove_stop_words(t) for t in words_in_tweets]
    stemmed_words = [stem_words(t) for t in stopwords_removed]
    formatted_tweets = [' '.join(split_tweet) for split_tweet in stemmed_words]
    return formatted_tweets

tweets = format_tweets(tweets)

uncommon_words_to_remove = find_uncommon_words(tweets)
tweets = remove_uncommon_words(tweets, uncommon_words_to_remove)

print(f'First 5 formatted tweets with uncommon words removed: \n{tweets[:5]}')

tweet_df = pd.DataFrame(tweets)
tweet_df.columns = ['tweets'] 

print(f'First 5 rows of new dataframe: \n{tweet_df.head()}') 

# Check number of unique tweets now that 'rt's have been removed.
print(f'Number of unique tweets: {tweet_df.tweets.nunique()}') 

# Save the csv of processed tweets to open in the neural net notebook
tweet_df.to_csv('./df_for_gensim.csv', index=False)

print("Formatted tweets saved to df_for_gensim.csv")
