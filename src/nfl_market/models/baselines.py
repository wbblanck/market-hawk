from sklearn.dummy import DummyClassifier,DummyRegressor
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
def classifiers():
    return {"prior":DummyClassifier(strategy="prior"),"logistic":make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000))}
def regressors():
    return {"mean":DummyRegressor(strategy="mean"),"ridge":make_pipeline(StandardScaler(),Ridge(alpha=10))}
