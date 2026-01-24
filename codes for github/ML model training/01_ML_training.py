# -*- coding: utf-8 -*-
"""
Created on Sat Jan 24 11:22:28 2026

@author: Kaiwei Luo
"""

# -*- coding: utf-8 -*-
"""
Train fire potential models (ABD/OBE) and save artifacts.
- Supports Hierarchical Structure: OBE model trained on fire-active samples only.
- Thresholds chosen by maximizing F1-score on Precision-Recall curve.
"""

import os, time, json, warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedStratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, accuracy_score, f1_score,
    precision_score, recall_score, precision_recall_curve, classification_report
)
from imblearn.under_sampling import RandomUnderSampler

warnings.filterwarnings("ignore")

# ----------------------------
# 0) Configuration & Paths
# ----------------------------
# Set your local directories here
MODEL_DIR = "./output_models"
CSV_PATH = "./data/fire_weather_training_data.csv"

os.makedirs(MODEL_DIR, exist_ok=True)

# Load dataset
df_train = pd.read_csv(CSV_PATH)
print(f"Initial training data shape: {df_train.shape}")

# ----------------------------
# 1) Variable Definition
# ----------------------------
target_obe = 'OBE'
target_h24 = 'h24event'
target_ratio = 'dailyspan_ratio'

feature_vars = ['BUI', 'DMC', 'DC', 'FWI', 'FFMC', 'ISI']
categorical_vars = ['biome', 'month']

# Drop missing values in feature/categorical columns
df_train = df_train.dropna(subset=feature_vars + categorical_vars)
print(f"Shape after dropping missing values: {df_train.shape}")

# Define fixed categories to lock One-Hot column order
all_biomes = [11, 12, 13, 16, 21, 22, 23, 24, 25, 31, 32, 33, 34, 35, 41, 42, 43, 50, 90]
all_months = list(range(1, 13))

# ----------------------------
# 2) Preprocessing (OHE & Scaling)
# ----------------------------
encoder = OneHotEncoder(
    sparse_output=False, 
    drop=None,
    categories=[all_biomes, all_months],
    handle_unknown='ignore'
)

# Fit One-Hot Encoding
onehot = encoder.fit_transform(df_train[categorical_vars])
onehot_cols = encoder.get_feature_names_out(categorical_vars)

# Build feature matrix: Numerical + One-Hot
X = pd.concat(
    [df_train[feature_vars].reset_index(drop=True),
     pd.DataFrame(onehot, columns=onehot_cols)],
    axis=1
)

# Standardize only numerical features
scaler = StandardScaler()
X[feature_vars] = scaler.fit_transform(X[feature_vars])

# Define target series
y_obe = df_train[target_obe].astype(int)
y_h24 = df_train[target_h24].astype(int)

# ABDp (Active Burning Day potential) defined as ratio > 0
df_train['has_fire'] = (df_train[target_ratio] > 0).astype(int)
y_has_fire = df_train['has_fire']

# Duration classification (Low vs High)
def categorize_duration(ratio):
    hours = ratio * 24.0
    return 0 if hours <= 12 else 1

df_train['duration_class'] = df_train[target_ratio].apply(categorize_duration)
y_duration = df_train['duration_class'].astype(int)

# Align indices
X.index = range(len(X))
for s in (y_obe, y_h24, y_has_fire, y_duration):
    s.index = X.index

print(f"Preprocessing complete. Total features: {X.shape[1]}")

# ----------------------------
# 3) Sampling & Splitting
# ----------------------------
# HIERARCHICAL_MODE: 
# True  -> Train OBE/H24/Duration ONLY on fire-active days (Matches Manuscript Description)
# False -> Train on all samples (Original Logic)
HIERARCHICAL_MODE = True 

rus = RandomUnderSampler(sampling_strategy=1.0, random_state=42)

# ABD (has_fire) always trained on full dataset
X_fire_bal, y_fire_bal = rus.fit_resample(X, y_has_fire)

if HIERARCHICAL_MODE:
    print("Mode: Hierarchical (Conditional OBE training given fire activity)")
    mask_fire_active = (y_has_fire == 1)
    X_fire_only = X[mask_fire_active]
    
    # Train OBE and H24 only on days where fire is active
    X_obe_bal, y_obe_bal = rus.fit_resample(X_fire_only, y_obe[mask_fire_active])
    X_h24_bal, y_h24_bal = rus.fit_resample(X_fire_only, y_h24[mask_fire_active])
else:
    print("Mode: Standard (Independent training on full dataset)")
    X_obe_bal, y_obe_bal = rus.fit_resample(X, y_obe)
    X_h24_bal, y_h24_bal = rus.fit_resample(X, y_h24)

# Duration: Always trained on fire-active samples for balancing intensity classes
mask_fire = (y_has_fire == 1)
X_fire_only = X[mask_fire]
y_dur_fire = y_duration[mask_fire]
min_cnt = y_dur_fire.value_counts().min()
sel_idx = np.concatenate([y_dur_fire[y_dur_fire == c].sample(min_cnt, random_state=42).index
                          for c in y_dur_fire.unique()])
X_dur_bal = X_fire_only.loc[sel_idx]
y_dur_bal = y_dur_fire.loc[sel_idx]

# Split function
def split_data(X_, y_):
    return train_test_split(X_, y_, test_size=0.2, random_state=42, stratify=y_)

Xtr_fire, Xte_fire, ytr_fire, yte_fire = split_data(X_fire_bal, y_fire_bal)
Xtr_obe,  Xte_obe,  ytr_obe,  yte_obe  = split_data(X_obe_bal,  y_obe_bal)
Xtr_h24,  Xte_h24,  ytr_h24,  yte_h24  = split_data(X_h24_bal,  y_h24_bal)
Xtr_dur,  Xte_dur,  ytr_dur,  yte_dur  = split_data(X_dur_bal,  y_dur_bal)

# ----------------------------
# 4) Evaluation Helper Functions
# ----------------------------
def eval_and_find_threshold(model, X_, y_, name):
    """Calculate metrics and find optimal threshold by maximizing F1-score."""
    proba = model.predict_proba(X_)[:, 1]
    prec, rec, thr = precision_recall_curve(y_, proba)
    f1s = 2 * prec * rec / (prec + rec + 1e-10)
    best_i = np.argmax(f1s[:-1]) if len(f1s) > 1 else 0
    best_t = thr[best_i] if len(thr) > 0 else 0.5
    
    pred = (proba >= best_t).astype(int)
    results = dict(
        auc=roc_auc_score(y_, proba),
        acc=accuracy_score(y_, pred),
        f1=f1_score(y_, pred),
        pre=precision_score(y_, pred, zero_division=0),
        rec=recall_score(y_, pred, zero_division=0),
        thr=best_t,
        cm=confusion_matrix(y_, pred)
    )
    print(f"[{name}] AUC: {results['auc']:.4f} | F1: {results['f1']:.4f} | Threshold: {results['thr']:.3f}")
    return results

def train_with_cv(candidates, Xtr, ytr, Xte, yte, task_name):
    """Perform CV to pick best model, then evaluate on test set."""
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    best_auc = -np.inf
    best_mdl = None

    for name, mdl in candidates:
        scores = cross_val_score(clone(mdl), Xtr, ytr, cv=cv, scoring='roc_auc', n_jobs=-1)
        print(f"[CV] {task_name}-{name}: AUC Mean={scores.mean():.4f}")
        if scores.mean() > best_auc:
            best_auc = scores.mean()
            best_mdl = mdl

    final_model = clone(best_mdl)
    final_model.fit(Xtr, ytr)
    metrics = eval_and_find_threshold(final_model, Xte, yte, task_name)
    return final_model, metrics['thr'], metrics

# ----------------------------
# 5) Training Execution
# ----------------------------
LR = LogisticRegression(C=0.1, max_iter=1000, solver='liblinear', random_state=42)
RF = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)

# ABDp (Fire Activity) Model
model_fire, thr_fire, m_fire = train_with_cv([("LR", LR), ("RF", RF)], Xtr_fire, ytr_fire, Xte_fire, yte_fire, "ABDp")

# OBEp (Overnight) Model
model_obe, thr_obe, m_obe = train_with_cv([("LR", LR), ("RF", RF)], Xtr_obe, ytr_obe, Xte_obe, yte_obe, "OBEp")

# H24 Event Model
model_h24, thr_h24, m_h24 = train_with_cv([("LR", LR), ("RF", RF)], Xtr_h24, ytr_h24, Xte_h24, yte_h24, "H24_Event")

# Duration Class (RF with balanced weights)
model_dur = RandomForestClassifier(n_estimators=500, class_weight='balanced', random_state=42, n_jobs=-1)
model_dur.fit(Xtr_dur, ytr_dur)
print("\n[Duration Classification Report]")
print(classification_report(yte_dur, model_dur.predict(Xte_dur), target_names=['Low(<=12h)', 'High(>12h)']))

# ----------------------------
# 6) Saving Artifacts
# ----------------------------
joblib.dump(encoder,   os.path.join(MODEL_DIR, 'encoder.pkl'))
joblib.dump(scaler,    os.path.join(MODEL_DIR, 'scaler.pkl'))
joblib.dump(model_fire, os.path.join(MODEL_DIR, 'fire_model.pkl'))
joblib.dump(model_obe,  os.path.join(MODEL_DIR, 'obe_model.pkl'))
joblib.dump(model_h24,  os.path.join(MODEL_DIR, 'h24_model.pkl'))
joblib.dump(model_dur,  os.path.join(MODEL_DIR, 'duration_model.pkl'))
joblib.dump({'fire_threshold': thr_fire, 'obe_threshold': thr_obe, 'h24_threshold': thr_h24},
            os.path.join(MODEL_DIR, 'thresholds.pkl'))

# Save Summary Table
summary = pd.DataFrame([
    {'Task': 'ABDp', 'Model': type(model_fire).__name__, 'AUC': m_fire['auc'], 'F1': m_fire['f1'], 'Threshold': thr_fire},
    {'Task': 'OBEp', 'Model': type(model_obe).__name__,  'AUC': m_obe['auc'],  'F1': m_obe['f1'],  'Threshold': thr_obe},
    {'Task': 'H24',  'Model': type(model_h24).__name__,  'AUC': m_h24['auc'],  'F1': m_h24['f1'],  'Threshold': thr_h24},
])
summary.to_csv(os.path.join(MODEL_DIR, "metrics_summary.csv"), index=False)

print(f"\nTraining complete. Models and metrics saved to: {MODEL_DIR}")