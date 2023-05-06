# machine learning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline


def get_dummy_model():
    # random chance baseline model
    model = DummyClassifier(
        strategy='stratified',
        random_state=42,
    )
    return model


def _get_vectoriser():
    # shared vectoriser
    _vectoriser = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 1),  # text_clean already includes ngrams
    )
    return _vectoriser


def get_naive_model():
    # naive bayes
    vectoriser = _get_vectoriser()
    model = ComplementNB(fit_prior=True,)
    pipe = Pipeline(
        steps=[('vectoriser', vectoriser), ('clf', model)],
        memory='cache',
    )
    return pipe


def get_linear_model():
    # linear models (logistic regression and svm)
    vectoriser = _get_vectoriser()
    model = SGDClassifier(
        penalty='l2',
        max_iter=1000,
        shuffle=True,
        n_jobs=-1,
        random_state=42,
        learning_rate='optimal',
        eta0=0.01,
        power_t=0.5,
    )
    pipe = Pipeline(
        steps=[('vectoriser', vectoriser), ('clf', model)],
        memory='cache',
    )
    return pipe
