from load_model import *
from view_model import *
from datetime import datetime
import timeit




def predict_lr(text):
    # start_time = datetime.now().second
    start = timeit.default_timer()

    vectorizer, model = load_model_unbiased(LogRegEnum.vectorizer_unbiased, LogRegEnum.model_unbiased)

    vec = vectorizer.transform([text])
    is_toxic = model.predict(vec)
    is_toxic_text = 'Toxic' if is_toxic[0] == 1 else 'Not toxic'
    probability = model.predict_proba(vec)[:, 1]

    stop = timeit.default_timer()
    execution_time = stop - start
    # end_time = datetime.now().second
    prediction_time = f"The prediction time in seconds : {round(execution_time,2)}"
    lr_model_unbiased = Model(model_name="Logistic regression unbiased",
                              probability=round(probability[0], 2),
                              is_toxic=is_toxic_text, prediction_time = prediction_time)

    return lr_model_unbiased
