import os
import pandas as pd

# Load the dataset
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(DATA_DIR, 'TrackD_Mental.csv'))


# Step 5: Object-Oriented Representation
class DataRecord:
    def __init__(self, record_id, features, label):
        self.record_id = record_id
        self.features = features
        self.label = label

    def display(self):
        print(f"Record ID: {self.record_id}")
        print("Features:")
        for key, value in self.features.items():
            print(f"  {key}: {value}")
        print(f"Label (treatment): {self.label}")
        print("-" * 40)


# Create 5 DataRecord objects from the first 5 rows
records = []
for i in range(5):
    row = df.iloc[i]
    # Store all columns except 'treatment' as features
    features = {}
    for col in df.columns:
        if col != 'treatment':
            features[col] = row[col]
    label = row['treatment']
    record = DataRecord(record_id=i, features=features, label=label)
    records.append(record)

# Display each record
print("======== DataRecord Objects ========\n")
for record in records:
    record.display()
