import os
import pandas as pd

# Load the dataset
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(DATA_DIR, 'TrackD_Mental.csv'))


# Step 6: Build a Graph from the Dataset
# Two meaningful columns: 'work_interfere' and 'treatment'
# Nodes = unique values from both columns
# Edge = two values appear together in the same row

col_a = 'work_interfere'
col_b = 'treatment'

# Drop rows where either column has missing values
subset = df[[col_a, col_b]].dropna()

# Get unique values from each column
nodes_a = subset[col_a].unique().tolist()
nodes_b = subset[col_b].unique().tolist()
all_nodes = list(set(nodes_a + nodes_b))

# Build the graph as an adjacency list (dictionary)
graph = {}
for node in all_nodes:
    graph[node] = []

for row in subset.itertuples(index=False):
    val_a = row.work_interfere
    val_b = row.treatment
    # Add edge if not already present
    if val_b not in graph[val_a]:
        graph[val_a].append(val_b)
    if val_a not in graph[val_b]:
        graph[val_b].append(val_a)

# Print the graph
print("======== Graph (Adjacency List) ========\n")
for node, neighbors in graph.items():
    print(f"  {node} --> {neighbors}")

# Count nodes and edges
total_nodes = len(all_nodes)
total_edges = 0
for node in graph:
    total_edges += len(graph[node])
total_edges = total_edges // 2  # each edge counted twice

print(f"\nTotal Nodes: {total_nodes}")
print(f"Total Edges: {total_edges}")
