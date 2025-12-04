import os
import joblib
import pandas as pd

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'rf_pipeline.joblib')


def load_model(path=MODEL_PATH):
    return joblib.load(path)


def predict_single(model, input_dict):
    # input_dict should map feature name -> value, matching training features
    row = pd.DataFrame([input_dict])

    proba = model.predict_proba(row)[0, 1]
    pred = int(model.predict(row)[0])
    return {'prediction': pred, 'probability': float(proba)}
