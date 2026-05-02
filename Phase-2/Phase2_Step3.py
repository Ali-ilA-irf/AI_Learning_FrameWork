import sys
import os
import heapq


from Phase2_Step1 import graph, agent

# For the heuristic, we'll need the features from Phase 1 Step 5
from Phase1_Step5 import records

# -----------------------------------------------------------------------------
# 1. HEURISTIC FUNCTION
# -----------------------------------------------------------------------------
# We define a "Goal Profile" from a record that sought treatment (Record 0)
goal_record = records[0]
goal_features = goal_record.features

def h(state):
    """
    Heuristic: Estimates distance to goal ('Yes').
    If the state is 'Yes', h=0.
    Otherwise, we check if the feature-value (state) matches the goal profile.
    If it matches, h=0 (likely close). If it doesn't match, h=1 (mismatch).
    """
    if state == 'Yes':
        return 0
    if state == 'No':
        return 2 # Farther away
    
    # Check if this feature value is in our goal profile
    # (Since our graph nodes are values like 'Often', 'Rarely')
    for val in goal_features.values():
        if str(val) == state:
            return 0  # Matches a 'Yes' profile feature
    return 1 # Mismatch

# -----------------------------------------------------------------------------
# 2. GREEDY BEST-FIRST SEARCH
# -----------------------------------------------------------------------------
def greedy_best_first_search(graph, start, goal):
    pq = [(h(start), start, [start])]
    visited = set()
    nodes_explored = 0

    while pq:
        _, current, path = heapq.heappop(pq)
        nodes_explored += 1

        if current == goal:
            return path, nodes_explored

        if current in visited:
            continue
        visited.add(current)

        for neighbor in graph.get(current, []):
            if neighbor not in visited:
                heapq.heappush(pq, (h(neighbor), neighbor, path + [neighbor]))

    return None, nodes_explored

# -----------------------------------------------------------------------------
# 3. A* SEARCH
# -----------------------------------------------------------------------------
def a_star_search(graph, start, goal):
    # pq stores: (f_score, current_node, path, g_score)
    # f = g + h
    pq = [(h(start), start, [start], 0)]
    visited = {} # node: g_score
    nodes_explored = 0

    while pq:
        f, current, path, g = heapq.heappop(pq)
        nodes_explored += 1

        if current == goal:
            return path, nodes_explored, g

        if current in visited and visited[current] <= g:
            continue
        visited[current] = g

        for neighbor in graph.get(current, []):
            new_g = g + 1 # uniform cost
            new_f = new_g + h(neighbor)
            heapq.heappush(pq, (new_f, neighbor, path + [neighbor], new_g))

    return None, nodes_explored, float('inf')

# -----------------------------------------------------------------------------
# 4. ADMISSIBILITY CHECK
# -----------------------------------------------------------------------------
def verify_admissibility(graph, goal):
    print("\n--- Admissibility Check (h(s) <= true_cost) ---")
    # Sample 5 states from the graph
    samples = ['Often', 'Sometimes', 'Rarely', 'Never', 'No']
    
    is_admissible = True
    for s in samples:
        # True cost is the shortest path found by BFS (since costs are 1)
        # We'll use a simple BFS to find actual shortest path
        from collections import deque
        def get_true_cost(start, goal):
            q = deque([(start, 0)])
            v = {start}
            while q:
                node, dist = q.popleft()
                if node == goal: return dist
                for n in graph.get(node, []):
                    if n not in v:
                        v.add(n)
                        q.append((n, dist + 1))
            return float('inf')

        true_cost = get_true_cost(s, goal)
        estimated = h(s)
        status = "PASS" if estimated <= true_cost else "FAIL"
        print(f"  State: {s:10} | h(s): {estimated} | true_cost: {true_cost} | {status}")
        if estimated > true_cost: is_admissible = False
    
    print(f"Heuristic is {'admissible' if is_admissible else 'NOT admissible'}.")
    return is_admissible
