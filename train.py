import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from model_utils import FeatureEngineer

RANDOM_STATE = 42
DATA_PATH = os.path.join('data', 'TelcoCustomerChurn.csv')
MODEL_DIR = 'model'
os.makedirs(MODEL_DIR, exist_ok=True)


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    return df


def make_preprocessor():
    numeric = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges', 'AvgMonthlyCharge', 'NumServices']
    categorical = [
        'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
        'PaperlessBilling', 'PaymentMethod', 'TenureGroup'
    ]
    return ColumnTransformer([
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), numeric),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                          ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical)
    ])


def metrics(y_true, y_pred, y_prob):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_prob),
        'pr_auc': average_precision_score(y_true, y_prob),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
    }


def main():
    df = load_data()
    y = (df['Churn'] == 'Yes').astype(int)
    X = df.drop(columns=['customerID', 'Churn'])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
    )

    def pipe(clf):
        return Pipeline([('features', FeatureEngineer()), ('prep', make_preprocessor()), ('clf', clf)])

    candidates = {
        'Decision Tree - baseline': pipe(DecisionTreeClassifier(random_state=RANDOM_STATE)),
        'Decision Tree - balanced': pipe(DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight='balanced')),
        'Logistic Regression - balanced': pipe(LogisticRegression(max_iter=1000, solver='liblinear', class_weight='balanced', random_state=RANDOM_STATE)),
        'Random Forest - balanced': pipe(RandomForestClassifier(n_estimators=250, random_state=RANDOM_STATE, class_weight='balanced', n_jobs=-1))
    }
    results = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1]
        results[name] = metrics(y_test, pred, prob)

    tuned = pipe(DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight='balanced'))
    grid = GridSearchCV(
        tuned,
        {
            'clf__max_depth': [3, 5, 7, 10],
            'clf__min_samples_leaf': [5, 10],
            'clf__min_samples_split': [2, 10]
        }, cv=5, scoring='f1', n_jobs=-1
    )
    grid.fit(X_train, y_train)
    best = grid.best_estimator_
    pred = best.predict(X_test)
    prob = best.predict_proba(X_test)[:, 1]
    results['Decision Tree - tuned + balanced'] = metrics(y_test, pred, prob)

    joblib.dump(best, os.path.join(MODEL_DIR, 'churn_pipeline.pkl'))
    with open(os.path.join(MODEL_DIR, 'metrics.json'), 'w') as f:
        json.dump({'best_params': grid.best_params_, 'cv_best_f1': grid.best_score_, 'test_results': results}, f, indent=2)

    # Save feature importances with names.
    feat = best.named_steps['prep'].get_feature_names_out()
    imp = best.named_steps['clf'].feature_importances_
    pd.DataFrame({'feature': feat, 'importance': imp}).sort_values('importance', ascending=False).to_csv(
        os.path.join(MODEL_DIR, 'feature_importance.csv'), index=False
    )
    print(json.dumps({'best_params': grid.best_params_, 'results': results}, indent=2))


if __name__ == '__main__':
    main()
