from typing import List

from load_model import *
from view_model import *
from scipy.sparse import hstack


def get_prediction(text) -> List[Model]:
    vectorizer, model = load_model_unbiased(LogRegEnum.vectorizer_unbiased, LogRegEnum.model_unbiased)

    vec = vectorizer.transform([text])
    is_toxic = model.predict(vec)
    is_toxic_text = 'Toxic' if is_toxic[0] == 1 else 'Not toxic'
    probability = model.predict_proba(vec)[:, 1]

    # vectorizer_words, vectorizer_chars, model_classifiers = \
    #     load_model_5classifiers(LogRegEnum.vectorizer_5classifiers_words, LogRegEnum.vectorizer_5classifiers_chars,
    #                             LogRegEnum.model_5classifiers)
    #
    # a = vectorizer_chars.transform([text])
    # b = vectorizer_words.transform([text])
    # stacked_vectorizer = hstack([a, b])
    # is_toxic_5classifiers = model_classifiers.predict(stacked_vectorizer)
    # probability_5_classifiers = model_classifiers.predict_proba(stacked_vectorizer)[:, 1]

    lr_model_unbiased = Model(model_name="Logistic regression unbiased",
                              probability=round(probability[0], 2),
                              is_toxic=is_toxic_text)

    # lr_model_5classifiers = Model(model_name="Logistic regression with 5 params",
    #                               probability=round(probability_5_classifiers[0], 2),
    #                               is_toxic=is_toxic_5classifiers)

    return [lr_model_unbiased]
