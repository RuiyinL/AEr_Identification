import nltk
import os
import re
import pandas as pd
import numpy as np
import string
from gensim.models import Word2Vec
from sklearn.preprocessing import normalize
from gensim.models.keyedvectors import KeyedVectors
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import fasttext
from preprocessing.w2vemb import EMB
from preprocessing.managedb import ManageDB
from preprocessing.preprocessing import preprocess
from nltk.corpus import stopwords
try:
    from torchtext.data import get_tokenizer
except Exception:
    get_tokenizer = None

# load SO_w2v_200
wv_from_bin = KeyedVectors.load_word2vec_format(
    os.path.expanduser("~/experiment/data/word_embedding/SO_vectors_200.bin"),
    binary=True,
)
# load FastText_200
fasttext.FastText.eprint = lambda x: None
ft = None
ft_word_dic = set()
try:
    ft = fasttext.load_model(os.path.expanduser('~/experiment/data/word_embedding/cc.en.100.bin'))
    ft_word_dic = set(ft.words)
except Exception:
    ft = None
    ft_word_dic = set()

# load GloVe_200
# gensim_file = '~/experiment/data/word_embedding/glove.twitter.27B.200d.txt'
# md = ManageDB()
# md.add_file2db('glove.twitter.27B.200d', gensim_file, 200, 1193513)
# GloVe = EMB(name='glove.twitter.27B.200d', dimensions=200)

read_path1 = r'~/experiment/data/Violation symptoms.xlsx'
read_path2 = r'~/experiment/data/Randomly_selected_comments.xlsx'

word2id = wv_from_bin.key_to_index  # dict: {word, index}; example: {'a': 0, 'b', 1, ...}
# ft_word_dic = ft.words

'''Feature Extraction'''
def get_word_vectors(embeding_model, word):
    if embeding_model == 'SO_w2v':
        if word == '0':
            word = '<UNK>'  # replace '0'
        if word in word2id:
            vector = wv_from_bin.get_vector(word)
        else:
            vector = np.array([0.] * 200, dtype=np.float64)

    if embeding_model == 'FastText':
        if word == '0':
            word = '<UNK>'
        if ft is not None and word in ft_word_dic:
            vector = ft.get_word_vector(word)
        else:
            vector = np.array([0.] * 300, dtype=np.float64)     # adjust the dimension

    if embeding_model == 'GloVe':
        if word == '0':
            word = '<UNK>'
        if word in GloVe:
            vector = np.array(GloVe.get_vector(word))
        else:
            vector = np.array([0.] * 200, dtype=np.float64)
    return vector

def get_sen_vectors(embeding_model, sentence):   # list; example: ['this', 'line', 'violat', 'new', 'hack', 'rule']
    # sentence_vec = np.array([0.] * 200, dtype=np.float64)
    sentence_vec = np.array([0.] * 300, dtype=np.float64)
    for word in sentence:
        sentence_vec += get_word_vectors(embeding_model, word)
    if len(sentence) > 0:
        sentence_vec = sentence_vec / float(len(sentence))
    return sentence_vec  # return: <class 'numpy.ndarray'>


def text_stats_features(texts):
    """Build lightweight text statistics features for each input string."""
    tokenizer = get_tokenizer("basic_english") if get_tokenizer is not None else None
    stop_words = set(stopwords.words("english"))
    punctuation = set(string.punctuation)
    rows = []
    for text in texts:
        if tokenizer is None:
            tokens = re.findall(r"[A-Za-z0-9_']+", text.lower())
        else:
            tokens = tokenizer(text)
        token_count = len(tokens)
        char_count = len(text)
        avg_token_len = (sum(len(t) for t in tokens) / token_count) if token_count else 0.0
        digit_count = sum(ch.isdigit() for ch in text)
        punct_count = sum(ch in punctuation for ch in text)
        upper_count = sum(ch.isupper() for ch in text)
        stop_count = sum(t in stop_words for t in tokens)
        stop_ratio = (stop_count / token_count) if token_count else 0.0
        upper_ratio = (upper_count / char_count) if char_count else 0.0
        rows.append([
            char_count,
            token_count,
            avg_token_len,
            digit_count,
            punct_count,
            stop_ratio,
            upper_ratio,
        ])
    return np.array(rows, dtype=np.float64)


def build_text_feature_matrix(texts, embeding_model, add_stats=True):
    """
    Combine sentence embeddings with extra text-stat features.
    Returns a numpy array of shape (n_samples, embedding_dim + extra_dim).
    """
    embedding_rows = []
    for text in texts:
        tokens = preprocess(text)
        embedding_rows.append(get_sen_vectors(embeding_model, tokens))
    embedding_matrix = np.vstack(embedding_rows)
    if not add_stats:
        return embedding_matrix
    stats_matrix = text_stats_features(texts)
    return np.hstack([embedding_matrix, stats_matrix])


if __name__ == '__main__':
    '''Manually choose the feature extraction method (i.e., SO_w2v, FastText, GloVe)'''
    data1 = pd.read_excel(os.path.expanduser(read_path1), sheet_name='combination', na_values='n/a')
    violation_comment = data1['Comment'].tolist()

    data2 = pd.read_excel(os.path.expanduser(read_path2), sheet_name='Comments', na_values='n/a')
    non_violation_comment = data2['Comment'].tolist()

    preprocessed_comment = []
    for item in violation_comment:
        # preprocessed_comment.append(get_sen_vectors('SO_w2v_200', preprocess(item)))
        preprocessed_comment.append(get_sen_vectors('FastText', preprocess(item)))
        # preprocessed_comment.append(get_sen_vectors('GloVe_200', preprocess(item)))
    Preprocessed_Data = pd.DataFrame(preprocessed_comment)

    # Preprocessed_Data.to_csv(r'~/experiment/data/extracted_features/SO_w2v_200_violation.csv', index=False)   # SO_w2v_200
    # Preprocessed_Data.to_csv(r'~/experiment/data/extracted_features/FastText_200_violation.csv', index=False)
    # Preprocessed_Data.to_csv(r'~/experiment/data/extracted_features/FastText_100_violation.csv', index=False)
    Preprocessed_Data.to_csv(r'~/experiment/data/extracted_features/FastText_300_violation.csv', index=False)
    # Preprocessed_Data.to_csv(r'~/experiment/data/extracted_features/GloVe_200_violation.csv', index=False)

    # irrelevant data
    preprocessed_comment = []
    for item in non_violation_comment:
        # preprocessed_comment.append(get_sen_vectors('SO_w2v_200', preprocess(item)))
        preprocessed_comment.append(get_sen_vectors('FastText', preprocess(item)))
        # preprocessed_comment.append(get_sen_vectors('GloVe_200', preprocess(item)))
    Preprocessed_Data = pd.DataFrame(preprocessed_comment)

    # Preprocessed_Data.to_csv(r'~/experiment/data/extracted_features/SO_w2v_200_non_violation.csv', index=False)
    # Preprocessed_Data.to_csv(r'~/experiment/data/extracted_features/FastText_200_non_violation.csv', index=False)
    # Preprocessed_Data.to_csv(r'~/experiment/data/extracted_features/FastText_100_non_violation.csv', index=False)
    Preprocessed_Data.to_csv(r'~/experiment/data/extracted_features/FastText_300_non_violation.csv', index=False)
    # Preprocessed_Data.to_csv(r'~/experiment/data/extracted_features/GloVe_200_non_violation.csv', index=False)
