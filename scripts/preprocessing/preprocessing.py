import re
import nltk
import string
import pandas as pd
import numpy as np
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from collections import defaultdict
from sklearn.pipeline import Pipeline
from gensim.models import Word2Vec
from sklearn.preprocessing import normalize

try:
    from torchtext.data import get_tokenizer
except Exception:
    get_tokenizer = None

'''Preprocessing'''
def recovery(tokens):  # list -> string
    recovey_sentence = ''
    for token in tokens:
        recovey_sentence = recovey_sentence + ' ' + token
    return recovey_sentence.strip()

def remove_url(text):
    text = text.replace('\r', '').replace('\n', '').replace('\t', '')
    text = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%|-|\#)*\b(\/)?', '', text, flags=re.MULTILINE)
    text = re.sub(r'(\/.*?\/)+.*? ', '', text, flags=re.MULTILINE)
    text = re.sub(r'(www\.)\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'  ', ' ', text, flags=re.MULTILINE)
    text = text.replace('\r', '').replace('\n', '').replace('\t', '')
    return text

# def tokenizer(text):
#     # create a tokenizer function
#     tokenizer = get_tokenizer("basic_english")
#     # text_list = []
#     # for i in text:
#     #     text_list.append(tokenizer(i))
#     # return text_list    # Return: [['xx',...,'x'], ..., ['xx',...,'x']]
#     return tokenizer(text)

def pad_or_cut(value: np.array, target_length: int):  # value: np.ndarray, target_length: int
    '''Pad or truncate a 1D numpy to a fixed-length numpy'''
    # value = np.array(value)
    data_row = None
    if len(value) < target_length:
        data_row = np.pad(value, (0, target_length - len(value)), 'constant', constant_values=int(0))
    elif len(value) > target_length:
        data_row = value[:target_length]
    return data_row

def preprocess(sentence):
    '''
    :param sentence: input a string
    :return: a token list of the string
    '''
    sentence = remove_url(sentence)     # remove the urls
    # token_words = nltk.word_tokenize(new_sentence)
    # token_words = nltk.wordpunct_tokenize(sentence)
    if get_tokenizer is None:
        token_words = re.findall(r"[A-Za-z0-9_']+", sentence.lower())
    else:
        tokenizer = get_tokenizer("basic_english")
        token_words = tokenizer(sentence)
    punctuation = list(set(string.punctuation))
    extra_special_characters = ["''", '``', '##','>>', 'e', 'g', 'eg', 'cant', 'cannot', 'isnt', 'would', 'could', 'doesnt', 'hasnt']
    special_characters = [c for c in extra_special_characters if c not in punctuation] + punctuation

    # Remove English stop words and special characters
    cleaned_words = [word for word in token_words if word not in stopwords.words('english')]    # Remove English stop words
    cleaned_words = [word for word in cleaned_words if word not in special_characters]  # Remove special characters
    cleaned_words = [word for word in cleaned_words if not word.isdigit()]  # Remove numbers
    cleaned_words = [word for word in cleaned_words if word.encode('utf-8').isalpha()]  # Remove special variable names
    # cleaned_words = [word.lower() for word in cleaned_words]    # Convert to low case

    # Stemming
    # Snowball_stemmer = SnowballStemmer('english')
    # stemmed_words = [Snowball_stemmer.stem(word) for word in cleaned_words]

    # Lemmatization
    # Lemmatizer = WordNetLemmatizer()
    # lemmatized_word = [Lemmatizer.lemmatize(word) for word in stemmed_words]
    # return recovery(lemmatized_word)  # string
    # return stemmed_words  # list
    return cleaned_words    # return: ['xx', 'xxx', ...]
