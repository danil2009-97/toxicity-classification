import pickle
from sklearn.linear_model import LogisticRegression
from enum import Enum


class LogRegEnum(str, Enum):
    vectorizer_unbiased = 'logreg_unbiased_vectorizer.pkl'
    model_unbiased = 'logreg_unbiased_model.pkl'
    vectorizer_5classifiers_words = 'logreg_5classifiers_vectorizer_word'
    vectorizer_5classifiers_chars = 'logreg_5classifiers_vectorizer_char'
    model_5classifiers= 'logreg_5classifiers_model.pkl'


def load_model_unbiased(vectorizer_path: LogRegEnum, model_path: LogRegEnum):
    with open(vectorizer_path, 'rb') as v:
        vectorizer = pickle.load(v)

    with open(model_path, 'rb') as w:
        model = pickle.load(w)

    return vectorizer, model


def load_model_5classifiers(vectorizer_words_path, vectorizer_chars_path, model_path):

    with open(vectorizer_words_path, 'rb') as vec_words:
       vectorizer_words  = pickle.load(vec_words)

    with open(vectorizer_chars_path, 'rb') as vec_chars:
        vectorizer_chars = pickle.load(vec_chars)


    with open(model_path, 'rb') as m:
        model = pickle.load(m)


    return vectorizer_words, vectorizer_chars, model



