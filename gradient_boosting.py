import re
from scipy.sparse import hstack
import lightgbm as lgb
import pickle
import string
import joblib
from load_model import *
from view_model import *
import timeit



def char_analyzer(text):
    """
    This is used to split strings in small lots
    I saw this in an article (I can't find the link anymore)
    so <talk> and <talking> would have <Tal> <alk> in common
    """
    tokens = text.split()
    return [token[i: i + 3] for token in tokens for i in range(len(token) - 2)]


cont_patterns = [
    (b'(W|w)on\'t', b'will not'),
    (b'(C|c)an\'t', b'can not'),
    (b'(I|i)\'m', b'i am'),
    (b'(A|a)in\'t', b'is not'),
    (b'(\w+)\'ll', b'\g<1> will'),
    (b'(\w+)n\'t', b'\g<1> not'),
    (b'(\w+)\'ve', b'\g<1> have'),
    (b'(\w+)\'s', b'\g<1> is'),
    (b'(\w+)\'re', b'\g<1> are'),
    (b'(\w+)\'d', b'\g<1> would'),
]

patterns = [(re.compile(regex), repl) for (regex, repl) in cont_patterns]


def prepare_for_char_n_gram(text):
    """ Simple text clean up process"""
    # 1. Go to lower case (only good for english)
    # Go to bytes_strings as I had issues removing all \n in r""
    clean = bytes(text.lower(), encoding="utf-8")
    # 2. Drop \n and  \t
    clean = clean.replace(b"\n", b" ")
    clean = clean.replace(b"\t", b" ")
    clean = clean.replace(b"\b", b" ")
    clean = clean.replace(b"\r", b" ")
    # 3. Replace english contractions
    for (pattern, repl) in patterns:
        clean = re.sub(pattern, repl, clean)
    # 4. Drop puntuation
    # I could have used regex package with regex.sub(b"\p{P}", " ")
    exclude = re.compile(b'[%s]' % re.escape(bytes(string.punctuation, encoding='utf-8')))
    clean = b" ".join([exclude.sub(b'', token) for token in clean.split()])
    # 5. Drop numbers - as a scientist I don't think numbers are toxic ;-)
    clean = re.sub(b"\d+", b" ", clean)
    # 6. Remove extra spaces - At the end of previous operations we multiplied space accurences
    clean = re.sub(b'\s+', b' ', clean)
    # Remove ending space if any
    clean = re.sub(b'\s+$', b'', clean)
    # 7. Now replace words by words surrounded by # signs
    # e.g. my name is bond would become #my# #name# #is# #bond#
    # clean = re.sub(b"([a-z]+)", b"#\g<1>#", clean)
    clean = re.sub(b" ", b"# #", clean)  # Replace space
    clean = b"#" + clean + b"#"  # add leading and trailing #

    return str(clean, 'utf-8')


def load_model_two_vectorizers(word_vec_path, char_vec_path, model_path):
    word_tokenizer = joblib.load(word_vec_path)
    char_tokenizer = joblib.load(char_vec_path)
    model = joblib.load(model_path)

    return word_tokenizer, char_tokenizer, model


def predict_lgbm(text):
    # Light GBM
    start = timeit.default_timer()

    word_tokenizer, char_tokenizer, lgbm_model = load_model_two_vectorizers(LgbmEnum.word_tokenizer,
                                                                            LgbmEnum.char_tokenizer,
                                                                            LgbmEnum.model_lgb)

    clean_comment = prepare_for_char_n_gram(text)
    clean_comment = [clean_comment]

    sample_char_vectorized = char_tokenizer.transform(clean_comment)
    sample_word_vectorized = word_tokenizer.transform(clean_comment)
    sample_sub = hstack(
        [
            sample_char_vectorized,
            sample_word_vectorized,
        ]).tocsr()

    is_toxic_lgbm = lgbm_model.predict(sample_sub, num_iteration=lgbm_model.best_iteration,
                                       predict_disable_shape_check=True)
    is_toxic_text = 'Toxic' if is_toxic_lgbm[0] >= 0.5 else 'Not toxic'

    stop = timeit.default_timer()
    execution_time = stop - start
    # end_time = datetime.now().second
    prediction_time = f"The prediction time in seconds : {round(execution_time, 2)}"


    light_gbm = Model(model_name="Light Gradient Boosting Machine (LGBM)", probability=round(is_toxic_lgbm[0], 2),
                      is_toxic=is_toxic_text, prediction_time = prediction_time)
    return light_gbm


