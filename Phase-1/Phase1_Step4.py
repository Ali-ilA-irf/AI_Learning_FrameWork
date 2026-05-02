import os
import pandas as pd
import numpy as np

# Load the dataset
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(DATA_DIR, 'TrackD_Mental.csv'))

# 1. Count rows per class using a for loop
class_counts = {}
for label in df['treatment']:
    if label in class_counts:
        class_counts[label] += 1
    else:
        class_counts[label] = 1

print("-------- Class Distribution (using loop) --------")
print(class_counts)

# 2. List of all column names
column_names = list(df.columns)
print("\n-------- All Column Names --------")
print(column_names)

# 3. First 50 rows stored in sample_data
sample_data = df.iloc[:50]
print("\n-------- Sample Data (first 50 rows) --------")
print(sample_data.shape)

# 4. describe_dataset() function
def describe_dataset():
    print("-------- Dataset Description --------")
    print("Shape:", df.shape)
    print("\nColumn Names:")
    for col in df.columns:
        print(" -", col)
    print("\nClass Distribution (target = 'treatment'):")
    for label, count in class_counts.items():
        print(f"  {label}: {count}")

describe_dataset()