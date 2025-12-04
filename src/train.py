import os
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'Flood_Prediction_NCR_Philippines.csv')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(MODEL_DIR, exist_ok=True)


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    return df


def preprocess_and_train(df):
    # normalize column names depending on dataset
    cols = set(df.columns.str.lower())
    if 'floodoccurrence' in df.columns or 'FloodOccurrence' in df.columns:
        # Uploaded dataset format
        df = df.rename(columns={
            'Rainfall_mm': 'rainfall',
            'WaterLevel_m': 'water_level',
            'SoilMoisture_pct': 'soil_moisture',
            'Elevation_m': 'elevation',
            'Location': 'location',
            'FloodOccurrence': 'flood'
        })
        # if Date column exists, convert
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        else:
            raise ValueError('Date column required for time-based split')

        # Use rainfall, elevation, and location for more signal
        features = ['rainfall', 'elevation', 'location']
        # Add random noise to rainfall to simulate measurement error
        df['rainfall'] = df['rainfall'] + np.random.normal(0, 2, size=len(df))
        # Add engineered feature: rainfall bin
        df['rainfall_bin'] = pd.cut(df['rainfall'], bins=[-1, 5, 15, 30, 1000], labels=['low','moderate','high','extreme'])
        features.append('rainfall_bin')
        X = df[features]
        y = df['flood']

        # Time-based split, but stratify by flood label within each period for more realistic class balance
        df_sorted = df.sort_values('Date')
        n = len(df_sorted)
        train_end = int(n * 0.7)
        val_end = int(n * 0.85)
        train_idx = df_sorted.index[:train_end]
        val_idx = df_sorted.index[train_end:val_end]
        test_idx = df_sorted.index[val_end:]
        # Optionally shuffle within each split to avoid blocks of only one class
        X_train, y_train = X.loc[train_idx], y.loc[train_idx]
        X_val, y_val = X.loc[val_idx], y.loc[val_idx]
        X_test, y_test = X.loc[test_idx], y.loc[test_idx]
        # Shuffle within splits
        X_train, y_train = X_train.sample(frac=1, random_state=42), y_train.sample(frac=1, random_state=42)
        X_val, y_val = X_val.sample(frac=1, random_state=42), y_val.sample(frac=1, random_state=42)
        X_test, y_test = X_test.sample(frac=1, random_state=42), y_test.sample(frac=1, random_state=42)
    else:
        # fallback to original small sample format
        df = df.rename(columns={
            'Temperature': 'temperature',
            'Humidity': 'humidity',
            'Rainfall': 'rainfall',
            'Wind Speed': 'wind_speed',
            'Pressure': 'pressure',
            'Location': 'location',
            'Flood': 'flood'
        })
        features = ['temperature', 'humidity', 'rainfall', 'wind_speed', 'pressure', 'location']
        X = df[features]
        y = df['flood']
        # fallback to random split
        strat = y if y.value_counts().min() >= 2 else None
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=strat)
        strat_temp = y_temp if y_temp.value_counts().min() >= 2 else None
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=strat_temp)

    # Use stratify only when each class has at least 2 samples
    strat = y if y.value_counts().min() >= 2 else None
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=strat)
    strat_temp = y_temp if y_temp.value_counts().min() >= 2 else None
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=strat_temp)

    # determine numeric vs categorical features from selected features
    numeric_features = [c for c in features if c not in ['location', 'rainfall_bin']]
    categorical_features = []
    if 'location' in features:
        categorical_features.append('location')
    if 'rainfall_bin' in features:
        categorical_features.append('rainfall_bin')

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    # Random Forest pipeline with class_weight='balanced' for imbalanced data
    rf = Pipeline(steps=[('preprocessor', preprocessor), ('clf', RandomForestClassifier(random_state=42, class_weight='balanced'))])

    param_grid = {
        'clf__n_estimators': [100, 200],
        'clf__max_depth': [10, 20, None],
        'clf__min_samples_leaf': [1, 2, 5]
    }

    grid = GridSearchCV(rf, param_grid, cv=3, n_jobs=-1, scoring='f1')
    grid.fit(X_train, y_train)

    best_pipeline = grid.best_estimator_

    # Find best threshold for F1/precision on validation set
    y_val_proba = best_pipeline.predict_proba(X_val)[:, 1]
    from sklearn.metrics import precision_recall_curve
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_val_proba)
    f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = f1s.argmax()
    best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

    # Use best threshold for reporting
    y_val_pred = (y_val_proba >= best_thresh).astype(int)
    y_test_proba = best_pipeline.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_proba >= best_thresh).astype(int)

    metrics = {
        'accuracy': float(accuracy_score(y_val, y_val_pred)),
        'precision': float(precision_score(y_val, y_val_pred, zero_division=0)),
        'recall': float(recall_score(y_val, y_val_pred, zero_division=0)),
        'f1': float(f1_score(y_val, y_val_pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_val, y_val_proba)),
        'best_threshold': float(best_thresh)
    }

    test_metrics = {
        'accuracy': float(accuracy_score(y_test, y_test_pred)),
        'precision': float(precision_score(y_test, y_test_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_test_pred, zero_division=0)),
        'f1': float(f1_score(y_test, y_test_pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_test, y_test_proba)),
        'best_threshold': float(best_thresh)
    }

    # evaluate on validation
    y_pred = best_pipeline.predict(X_val)
    y_proba = best_pipeline.predict_proba(X_val)[:, 1]

    metrics = {
        'accuracy': float(accuracy_score(y_val, y_pred)),
        'precision': float(precision_score(y_val, y_pred, zero_division=0)),
        'recall': float(recall_score(y_val, y_pred, zero_division=0)),
        'f1': float(f1_score(y_val, y_pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_val, y_proba))
    }

    # final evaluation on test set
    y_test_pred = best_pipeline.predict(X_test)
    y_test_proba = best_pipeline.predict_proba(X_test)[:, 1]

    test_metrics = {
        'accuracy': float(accuracy_score(y_test, y_test_pred)),
        'precision': float(precision_score(y_test, y_test_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_test_pred, zero_division=0)),
        'f1': float(f1_score(y_test, y_test_pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_test, y_test_proba))
    }

    # Save model and metrics
    model_path = os.path.join(MODEL_DIR, 'rf_pipeline.joblib')
    joblib.dump(best_pipeline, model_path)

    metrics_path = os.path.join(MODEL_DIR, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump({'validation': metrics, 'test': test_metrics}, f, indent=2)

    print('Saved model to', model_path)
    print('Validation metrics:', metrics)
    print('Test metrics:', test_metrics)

    return best_pipeline, metrics, test_metrics


def main():
    df = load_data()
    preprocess_and_train(df)


if __name__ == '__main__':
    main()
