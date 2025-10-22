from sklearn.metrics import cohen_kappa_score, f1_score


def qwk(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro")
