import joblib
import re
from sklearn.base import BaseEstimator

def score(text: str, model: BaseEstimator, threshold: float) -> tuple:
    """Score text with spam detection"""
    vectorizer = joblib.load("vectorizer.pkl")
    text_tfidf = vectorizer.transform([text])
    
    try:
        propensity = model.predict_proba(text_tfidf)[0, 1]
    except AttributeError:
        # For models without predict_proba
        propensity = float(model.predict(text_tfidf)[0])
    
    prediction = propensity >= threshold
    return bool(prediction), float(propensity)