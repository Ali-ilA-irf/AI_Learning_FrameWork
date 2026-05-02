# Central printing script for Phase 2 Search Algorithms
from Phase2_Step2 import graph, bfs, dfs, dls, ids, ucs
from Phase2_Step3 import greedy_best_first_search, a_star_search, verify_admissibility
from Phase2_Step4 import run_hill_climbing_multiple_times, simulated_annealing, local_beam_search
from Phase2_Step5 import minimax, alphabeta

# Starting state: a person who says work often interferes with mental health
start = 'Often'
goal  = 'Yes'   # goal: sought treatment

print("-" * 60)
print("PHASE 2: SEARCH ALGORITHMS PERFORMANCE")
print("-" * 60)
print(f"Start Node: {start}")
print(f"Goal Node:  {goal}")

# =============================================================================
# UNINFORMED SEARCH
# =============================================================================
print("\n[UNINFORMED SEARCH]")

# --- BFS ---
path_bfs, nodes_bfs = bfs(graph, start, goal)
print(f"  BFS:  Path={path_bfs}, Explored={nodes_bfs}")

# --- DFS ---
path_dfs, nodes_dfs = dfs(graph, start, goal)
print(f"  DFS:  Path={path_dfs}, Explored={nodes_dfs}")

# --- IDS ---
path_ids, nodes_ids = ids(graph, start, goal)
print(f"  IDS:  Path={path_ids}, Explored={nodes_ids}")

# --- DL3 ---
path_dl3, nodes_dl3 = dls(graph, start, goal,limit=3)
print(f"  DL3: Path={path_dl3}, Explored={nodes_dl3}")

# --- DL5 ---
path_dl5, nodes_dl5 = dls(graph, start, goal, limit=5)
print(f"  DL5: Path={path_dl5}, Explored={nodes_dl5}")


# --- UCS ---
path_ucs, nodes_ucs, cost_ucs = ucs(graph, start, goal)
print(f"  UCS:  Path={path_ucs}, Explored={nodes_ucs}, Cost={cost_ucs}")

# =============================================================================
# INFORMED SEARCH
# =============================================================================
print("\n[INFORMED SEARCH]")

# --- Greedy Best-First ---
path_gbfs, nodes_gbfs = greedy_best_first_search(graph, start, goal)
print(f"  Best-First: Path={path_gbfs}, Explored={nodes_gbfs}")

# --- A* Search ---
path_astar, nodes_astar, cost_astar = a_star_search(graph, start, goal)
print(f"  A* Search:  Path={path_astar}, Explored={nodes_astar}, Cost={cost_astar}")

# --- Admissibility ---
verify_admissibility(graph, goal)

# =============================================================================
# BEYOND CLASSICAL SEARCH
# =============================================================================
print("\n[BEYOND CLASSICAL SEARCH]")

# --- Hill Climbing ---
hc_results, global_opts = run_hill_climbing_multiple_times(graph, num_runs=10)
print(f"  Hill Climbing: Reached global optimum (Yes) {global_opts}/10 times.")

# --- Simulated Annealing ---
sa_path, sa_end = simulated_annealing(graph, start)
print(f"  Simulated Annealing: Final State={sa_end}, Path Length={len(sa_path)}")

# --- Local Beam Search ---
beam_3 = local_beam_search(graph, k=3)
beam_5 = local_beam_search(graph, k=5)
print(f"  Local Beam (k=3): Final States={beam_3}")
print(f"  Local Beam (k=5): Final States={beam_5}")

# =============================================================================
# ADVERSARIAL SEARCH
# =============================================================================
print("\n[ADVERSARIAL SEARCH]")

depth = 4
nodes_minimax = [0]
nodes_ab = [0]

# --- Minimax ---
val_mm, move_mm = minimax(graph, start, depth, True, counter=nodes_minimax)
print(f"  Minimax: Best Value={val_mm}, Best Move={move_mm}, Nodes Evaluated={nodes_minimax[0]}")

# --- Alpha-Beta Pruning ---
val_ab, move_ab = alphabeta(graph, start, depth, -float('inf'), float('inf'), True, counter=nodes_ab)
print(f"  Alpha-Beta: Best Value={val_ab}, Best Move={move_ab}, Nodes Evaluated={nodes_ab[0]}")

print(f"  Pruning eliminated {nodes_minimax[0] - nodes_ab[0]} nodes compared to Minimax.")
print("-" * 60)
