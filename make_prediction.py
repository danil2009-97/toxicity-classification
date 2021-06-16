from typing import List
from gradient_boosting import *
from load_model import *
from view_model import *
from scipy.sparse import hstack
from logistic_regression import *


def get_prediction(text) -> List[Model]:
    logistic_reg = predict_lr(text)

    light_gbm = predict_lgbm(text)

    return [logistic_reg, light_gbm]

