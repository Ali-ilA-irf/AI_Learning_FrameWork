# AI-Driven Analysis of Mental Health in Tech

This project is a comprehensive Artificial Intelligence pipeline built to analyze the OSMI Mental Health in Tech Survey dataset. It implements fundamental AI paradigms from scratch, including Search Algorithms, Constraint Satisfaction Problems (CSPs), and Machine Learning models.

## 📂 Project Architecture & Modularity

The codebase is highly modular, designed to separate core algorithm logic from execution and output. The project is divided into four main phases, each containing individual Python scripts for specific steps outlined in the lab manual.

### How Printing & Execution is Handled
To prevent terminal spam and ensure a clean execution flow:
1. **Silenced Step Files:** Every individual algorithm script (e.g., `Phase2_Step3.py`) has its execution and printing logic wrapped inside an `if __name__ == '__main__':` block. This means the algorithms can be safely imported without accidentally running or printing anything.
2. **Phase Printing Hubs:** Each phase folder contains a central printing script (e.g., `Phase3_Printing.py`). These scripts import the silenced algorithms, execute them in order, and format the results into clean, readable console outputs and comparison tables.
3. **Master Hub:** The root directory contains a master `main.py` script. This script orchestrates the entire project by sequentially triggering the printing hubs of all four phases in isolated subprocesses.

## 🚀 How to Run the Project

To execute the entire AI pipeline from start to finish and view all outputs:

1. Open your terminal in the root directory (`AI_Learning_FrameWork`).
2. Run the master script:
   ```bash
   python main.py
   ```
3. *(Optional)* To save the massive output log to a text file for review, run:
   ```bash
   python main.py > Project_Output.txt
   ```

## 🧠 Phase Breakdown

### Phase 1: Environment & Data Representation
- **Step 3 & 4:** Dataset loading, inspection, and application of Python fundamentals.
- **Step 5:** Object-Oriented Programming (OOP) representation of the dataset using `DataRecord` objects.
- **Step 6:** Graph building. Parses survey columns (`work_interfere` and `treatment`) into an undirected state-space graph (adjacency list).

### Phase 2: Search Algorithms
Algorithms navigate the bipartite state-space graph built in Phase 1 to find paths from a feature state (e.g., `'Often'`) to a target state (e.g., `'Yes'`).
- **Uninformed Search:** Breadth-First Search (BFS), Depth-First Search (DFS), Depth-Limited Search (DLS limits 3 & 5), Iterative Deepening (IDS), Uniform Cost Search (UCS).
- **Informed Search:** Greedy Best-First Search, A* Search (using an admissible feature-mismatch heuristic).
- **Beyond Classical Search:** Hill Climbing, Simulated Annealing, Local Beam Search.
- **Adversarial Search:** Minimax and Alpha-Beta Pruning (modeled as a 2-player game predicting treatment).

### Phase 3: Constraint Satisfaction Problems (CSP)
Models employee profiles as a CSP to find consistent states without relying on the Phase 1 state-space graph.
- **Variables:** `work_interfere`, `benefits`, `care_options`, `seek_help`, `family_history`.
- **Constraint Propagation:** AC-3 Algorithm.
- **Backtracking:** Plain Backtracking, Forward Checking, and the Minimum Remaining Values (MRV) heuristic.
- **Local Search:** Min-Conflicts algorithm.

### Phase 4: Machine Learning
Predicts mental health treatment-seeking behavior using models built entirely from scratch (no Scikit-Learn models used).
- **Data Prep:** Missing value imputation, label encoding, numerical normalization, and an 80/20 train-test split.
- **Unsupervised Learning:** K-Means Clustering and K-Medoids Clustering (compared using Within-Cluster Distance).
- **Supervised Learning:** 
  - Perceptron (Single-layer, step activation)
  - Delta Rule / Adaline (Batch Gradient Descent minimizing MSE)
  - Multilayer Perceptron (MLP) (2 hidden layers, ReLU & Sigmoid activations, Backpropagation via Cross-Entropy Loss).

## 📄 Documentation
The final reflections, comparison tables, and the 1000-word research report are synthesized in `AI_Project.docx` and `Phase5_Research_Report.txt`.
