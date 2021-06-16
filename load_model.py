import pickle
from sklearn.linear_model import LogisticRegression
from enum import Enum
import joblib



class LogRegEnum(str, Enum):
    vectorizer_unbiased = 'logreg_unbiased_vectorizer.pkl'
    model_unbiased = 'logreg_unbiased_model.pkl'
    vectorizer_5classifiers_words = 'logreg_5classifiers_vectorizer_word'
    vectorizer_5classifiers_chars = 'logreg_5classifiers_vectorizer_char'
    model_5classifiers = 'logreg_5classifiers_model.pkl'


class LgbmEnum(str, Enum):
    word_tokenizer = 'word_vectorizer_model.pkl'
    char_tokenizer = 'char_vectorizer_lgbm.pkl'
    model_lgb = 'lgbm_model.pkl'


def load_model_unbiased(vectorizer_path: LogRegEnum, model_path: LogRegEnum):
    with open(vectorizer_path, 'rb') as v:
        vectorizer = pickle.load(v)

    with open(model_path, 'rb') as w:
        model = pickle.load(w)

    return vectorizer, model



