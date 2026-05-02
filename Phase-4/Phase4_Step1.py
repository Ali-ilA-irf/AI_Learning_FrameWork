import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

import os
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Phase-1'))
df = pd.read_csv(os.path.join(DATA_DIR, 'TrackD_Mental.csv'))

# ---------------------------------------------
# 1. HANDLE MISSING VALUES
# ---------------------------------------------
if __name__ == '__main__':
    print("\n--- Missing Values BEFORE Cleaning ---\n")
    print(df.isnull().sum())

# Fill missing values column by column safely
for col in df.columns:
    if df[col].isnull().sum() > 0:
        try:
            df[col] = df[col].fillna(df[col].mean())
        except:
            df[col] = df[col].fillna(df[col].mode()[0])

if __name__ == '__main__':
    print("\n---- Missing Values AFTER Cleaning ----\n")
    print(df.isnull().sum())

# ---------------------------------------------
# 2. DROP USELESS COLUMNS
# ---------------------------------------------
df = df.drop(columns=['Timestamp', 'comments'], errors='ignore')

# ---------------------------------------------
# 3. CONVERT EVERYTHING TO PLAIN PYTHON STRINGS
#    THEN ENCODE — this bypasses ArrowStringArray
# ---------------------------------------------
le = LabelEncoder()

if __name__ == '__main__':
    print("\n---- Encoding Categorical Columns ----\n")

for col in df.columns:
    # Convert column to numpy first to escape Arrow backend
    sample = df[col].iloc[0]
    try:
        float(sample)
        # If it converts to float, it is numeric — skip
    except (ValueError, TypeError):
        # It is a string column — encode it
        df[col] = le.fit_transform(
            df[col].astype(str).to_numpy()
        )
        if __name__ == '__main__':
            print(f"  {col} encoded")

# ---------------------------------------------
# 4. SEPARATE FEATURES AND TARGET
# ---------------------------------------------
X = df.drop(columns=['treatment'])
y = df['treatment']

if __name__ == '__main__':
    print("\n--- Features Shape ---", X.shape)
    print("--- Target Shape   ---", y.shape)
    print("--- Target classes ---", y.unique())

# ---------------------------------------------
# 5. NORMALIZE NUMERICAL FEATURES
# ---------------------------------------------

# Convert to numpy directly bypasses Arrow completely
X_array_raw = X.to_numpy(dtype=float, na_value=0.0)
y_array     = y.to_numpy().astype(int)

# Normalize
X_mean = X_array_raw.mean(axis=0)
X_std  = X_array_raw.std(axis=0)
X_std[X_std == 0] = 1
X_normalized = (X_array_raw - X_mean) / X_std

if __name__ == '__main__':
    print("\n---- Normalization Done ----\n")
    print(X_normalized[:3])

# ---------------------------------------------
# 6. TRAIN TEST SPLIT (80/20) — from scratch
# ---------------------------------------------

np.random.seed(42)

indices = np.arange(len(X_normalized))
np.random.shuffle(indices)

split     = int(0.8 * len(indices))
train_idx = indices[:split]
test_idx  = indices[split:]

X_train = X_normalized[train_idx]
X_test  = X_normalized[test_idx]
y_train = y_array[train_idx]
y_test  = y_array[test_idx]

if __name__ == '__main__':
    print("\n--- Train/Test Split ---")
    print("X_train shape:", X_train.shape)
    print("X_test shape :", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape :", y_test.shape)