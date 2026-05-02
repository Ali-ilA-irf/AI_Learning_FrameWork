import os
import pandas as pd
import numpy as np

# Load the dataset
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(DATA_DIR, 'TrackD_Mental.csv'))

# 1. First 10 rows
print("-------- First 10 Rows --------")
print(df.head(10))

# 2. Shape
print("\n-------- Shape (rows, columns) --------")
print(df.shape)

# 3. Data types
print("\n-------- Data Types --------")
print(df.dtypes)

# 4. Missing values
print("\n-------- Missing Values Per Column --------")
print(df.isnull().sum())

# 5. Target column — 'treatment' (Yes/No: did they seek mental health treatment?)
print("\n-------- Unique Values in Target Column (treatment) --------")
print(df['treatment'].value_counts())
print("Number of unique classes:", df['treatment'].nunique())