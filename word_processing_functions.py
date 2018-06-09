import string
import operator
import re
import nltk
from nltk.stem.snowball import EnglishStemmer
from stopwords import * 

other_stops = ['ff', 'rt', 'amp', 'youre', 'youve', 'youll', 'youd', 'shes', 'thatll', 
'dont', 'shouldve', 'aint', 'arent', 'couldnt', 'didnt', 'doesnt', 'hadnt', 'hasnt', 
'havent', 'isnt', 'mightnt', 'mustnt', 'neednt', 'shant', 'shouldnt', 'wasnt', 'werent', 
'wont', 'wouldnt']
stopwords.extend(other_stops)

def remove_punctuation(tweets_list):
    translator = str.maketrans('', '', string.punctuation)
    return [t.translate(translator) for t in tweets_list]

def remove_numbers(tweets_list):
    return [re.sub('\d+', '', t) for t in tweets_list]

def remove_stop_words(word_list):
    return [word for word in word_list if word not in stopwords]

def stem_words(word_list):
    stemmer = EnglishStemmer()
    return [stemmer.stem(w) for w in word_list]

def find_uncommon_words(tweets):
    combined_string_of_all_tweets = ' '.join(tweets)
    unique_words_sorted_by_occurrence = word_count(combined_string_of_all_tweets)
    return [t[0] for t in unique_words_sorted_by_occurrence if t[1] < 2]

def word_count(combined_string_of_all_tweets):
    counts = dict()
    words = combined_string_of_all_tweets.split() 

    for word in words:
        if word in counts:
            counts[word] += 1
        else:
            counts[word] = 1
    sorted_counts = sorted(counts.items(), key=operator.itemgetter(1))
    return sorted_counts

def remove_uncommon_words_from_tweet(tweet, uncommon_words):
    return [word for word in tweet if word not in uncommon_words]

def remove_uncommon_words(tweets, uncommon_words):
    words_in_tweets = [t.split() for t in tweets]
    uncommon_words_removed = [remove_uncommon_words_from_tweet(t, uncommon_words) for t in words_in_tweets]
    return [' '.join(split_tweet) for split_tweet in uncommon_words_removed]

# Note: The most common word in this data set, with over twice as many occurences than the second most common one, 
# is "bitch". The second most common one is "hoe". I hate this data set.