from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from tabpfn import TabPFNClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

# Veri setini yükle
data = load_iris()
X, y = data.data, data.target

models = {
    "TABPFN": TabPFNClassifier(device='cpu'),
    "RandomForest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(eval_metric='mlogloss'),
    "LightGBM": LGBMClassifier()
}

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    print(f"{name} Accuracy: {np.mean(scores) * 100:.2f}%")
