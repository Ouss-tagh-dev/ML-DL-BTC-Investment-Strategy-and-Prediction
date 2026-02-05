import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
import os
import json

FILE_PATH = r"c:\Users\15086\Desktop\ml-dl-dashbord\dl-ml-btc\data\features\btc_features_complete.csv"

def select_features(df, k=35):
    target = 'target_direction'
    # Exclude future columns
    X_raw = df.drop([c for c in df.columns if 'target' in c.lower() or 'future' in c.lower()], axis=1)
    y = df[target]
    
    # Univariate selection
    selector = SelectKBest(score_func=f_classif, k=min(k*2, X_raw.shape[1]))
    selector.fit(X_raw, y)
    top_raw_cols = X_raw.columns[selector.get_support()]
    
    # Correlation filter
    corr_matrix = X_raw[top_raw_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    final_cols = [c for c in top_raw_cols if c not in to_drop][:k]
    
    return final_cols

def select_features_nb(df, k=15):
    target = 'target_direction'
    X_raw = df.drop([c for c in df.columns if 'target' in c.lower() or 'future' in c.lower()], axis=1)
    y = df[target]
    
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X_raw, y)
    
    top_cols = X_raw.columns[selector.get_support()].tolist()
    return top_cols

def load_and_preprocess(path):
    print(f"Loading {path}...")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df.select_dtypes(include=[np.number])
    
    # Lags
    target_lags = ['returns', 'RSI_14', 'MACD', 'MACD_hist', 'Close', 'Volume']
    existing_lags = [f for f in target_lags if f in df.columns]
    for f in existing_lags:
        for lag in range(1, 6):
            df[f"{f}_lag_{lag}"] = df[f].shift(lag)
    return df.ffill().bfill().dropna()

if __name__ == "__main__":
    if not os.path.exists(FILE_PATH):
        print("Data file not found!")
    else:
        df = load_and_preprocess(FILE_PATH)
        
        print("\n--- SVM Features (k=35) ---")
        svm_feats = select_features(df, k=35)
        print(json.dumps(svm_feats))
        
        print("\n--- Naive Bayes Features (k=15) ---")
        nb_feats = select_features_nb(df, k=15)
        print(json.dumps(nb_feats))
