import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Leakage-safe feature engineering for the Telco churn dataset."""
    service_cols = [
        'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if 'TotalCharges' in X.columns:
            X['TotalCharges'] = pd.to_numeric(X['TotalCharges'], errors='coerce')
        if 'tenure' in X.columns and 'TotalCharges' in X.columns:
            X['AvgMonthlyCharge'] = X['TotalCharges'] / X['tenure'].replace(0, 1)
        present = [c for c in self.service_cols if c in X.columns]
        if present:
            X['NumServices'] = X[present].apply(lambda r: (r == 'Yes').sum(), axis=1)
        if 'tenure' in X.columns:
            X['TenureGroup'] = pd.cut(
                X['tenure'], bins=[-1, 6, 12, 24, 48, np.inf],
                labels=['0-6', '7-12', '13-24', '25-48', '49+']
            ).astype('object')
        return X
