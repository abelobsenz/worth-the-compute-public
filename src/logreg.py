"""Logistic regression with its settings chosen by cross-validation.

Used by three methods: TF-IDF + LR, BGE + LR (both in train_classic.py) and SetFit
(train_setfit.py). All three end with the same kind of classifier, so differences between them
come from how the text is turned into numbers, not from the classifier.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline

# The settings we try:
#   C             how closely the model may fit the training data (larger = closer fit)
#   class_weight  None treats every example alike; "balanced" gives rare classes more weight
SETTINGS_TO_TRY = {"C": [1, 10, 100], "class_weight": [None, "balanced"]}


def fit_logistic_regression(features, labels, seed, steps_before=None):
    """Train a logistic regression and pick its settings by 3-fold cross-validation.

    features      the training examples: raw texts when steps_before holds a vectorizer, or one
                  row of numbers per example
    labels        the class of every example
    steps_before  optional list of preprocessing steps (e.g. a TF-IDF vectorizer) that must be
                  fitted on the training folds only

    Cross-validation uses only the examples it is given. It splits them into 3 parts, trains on
    2 and scores the third, and repeats so every part is scored once. The setting with the best
    average macro-F1 is then trained on all the examples. Because nothing outside the budget is
    used, a run with 8 examples per class really only sees 8 labels per class.

    Returns the trained model and the chosen settings.
    """
    classifier = LogisticRegression(max_iter=2000)
    model = make_pipeline(*(steps_before or []), classifier)
    # Inside a pipeline, a setting is addressed as "<step name>__<setting name>".
    grid = {f"logisticregression__{name}": values for name, values in SETTINGS_TO_TRY.items()}

    # Each fold needs at least one example of every class, so the smallest class limits the
    # number of folds. With fewer than 2 folds we cannot cross-validate and use C=10.
    n_folds = min(3, np.bincount(labels).min())
    if n_folds < 2:
        return model.set_params(**{"logisticregression__C": 10}).fit(features, labels), {}

    folds = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
    search = GridSearchCV(model, grid, scoring="f1_macro", cv=folds, n_jobs=-1, refit=True)
    search.fit(features, labels)
    chosen = {name.split("__")[1]: value for name, value in search.best_params_.items()}
    return search.best_estimator_, chosen


def log_probabilities(model, features, n_classes):
    """The model's log-probability of every class, as a table with one column per class.

    A model only knows the classes it saw in training. If a class was missing, its column gets a
    very low score (-10000), so that class is never predicted.
    """
    scores = np.full((len(features), n_classes), -1e4, dtype=np.float32)
    probabilities = np.clip(model.predict_proba(features), 1e-12, None)  # avoid log(0)
    scores[:, model.classes_] = np.log(probabilities)
    return scores
