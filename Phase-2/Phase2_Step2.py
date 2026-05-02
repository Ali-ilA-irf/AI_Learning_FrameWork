import sys
import os
from collections import deque
import heapq

# Import agent and graph from Phase2_Step1
from Phase2_Step1 import agent, graph

if __name__ == '__main__':
    print("\nAIAgent and Graph imported from Phase2_Step1 successfully.")
    print("Goal state:", agent.goal)



# --------------------------------------------
# 1. BREADTH FIRST SEARCH (BFS)
# --------------------------------------------
def bfs(graph, start, goal):
    queue = deque()
    queue.append((start, [start]))  # (current_node, path_so_far)
    visited = set()
    nodes_explored = 0

    while queue:
        node, path = queue.popleft()
        nodes_explored += 1

        if node == goal:
            return path, nodes_explored

        if node in visited:
            continue
        visited.add(node)

        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                queue.append((neighbor, path + [neighbor]))

    return None, nodes_explored  # no path found

# --------------------------------------------
# 2. DEPTH FIRST SEARCH (DFS)
# --------------------------------------------
def dfs(graph, start, goal):
    stack = [(start, [start])]  # (current_node, path_so_far)
    visited = set()
    nodes_explored = 0

    while stack:
        node, path = stack.pop()
        nodes_explored += 1

        if node == goal:
            return path, nodes_explored

        if node in visited:
            continue
        visited.add(node)

        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                stack.append((neighbor, path + [neighbor]))

    return None, nodes_explored

# --------------------------------------------
# 3. DEPTH LIMITED SEARCH (DLS)
# --------------------------------------------
def dls(graph, start, goal, limit):
    def recursive_dls(node, path, depth, visited, counter):
        counter[0] += 1

        if node == goal:
            return path

        if depth == 0:
            return None  # depth limit reached

        visited.add(node)
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                result = recursive_dls(neighbor, path + [neighbor], depth - 1, visited.copy(), counter)
                if result is not None:
                    return result
        return None

    counter = [0]
    result = recursive_dls(start, [start], limit, set(), counter)
    return result, counter[0]

# --------------------------------------------
# 4. ITERATIVE DEEPENING SEARCH (IDS)
# --------------------------------------------
def ids(graph, start, goal, max_depth=10):
    total_nodes = 0
    for depth in range(max_depth + 1):
        path, nodes = dls(graph, start, goal, depth)
        total_nodes += nodes
        if path is not None:
            print(f"  Goal found at depth: {depth}")
            return path, total_nodes
    return None, total_nodes

# --------------------------------------------
# 5. UNIFORM COST SEARCH (UCS)
# ---------------------------------------------
def ucs(graph, start, goal):
    # priority queue: (cost, node, path)
    pq = [(0, start, [start])]
    visited = set()
    nodes_explored = 0

    while pq:
        cost, node, path = heapq.heappop(pq)
        nodes_explored += 1

        if node == goal:
            return path, nodes_explored, cost

        if node in visited:
            continue
        visited.add(node)

        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                new_cost = cost + 1  # uniform cost = 1 per edge
                heapq.heappush(pq, (new_cost, neighbor, path + [neighbor]))

    return None, nodes_explored, float('inf')