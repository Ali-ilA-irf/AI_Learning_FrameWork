import random
import math
from Phase2_Step1 import graph

# -----------------------------------------------------------------------------
# OBJECTIVE FUNCTION
# -----------------------------------------------------------------------------
# We define an objective value for each state to guide local search toward 'Yes'
state_values = {
    'Yes': 100,
    'Often': 80,
    'Sometimes': 60,
    'Rarely': 40,
    'Never': 20,
    'No': 0
}

def get_value(state):
    return state_values.get(state, 10)

# -----------------------------------------------------------------------------
# 1. HILL CLIMBING
# -----------------------------------------------------------------------------
def hill_climbing(graph, start):
    current = start
    path = [current]
    
    while True:
        neighbors = graph.get(current, [])
        if not neighbors:
            break
            
        # Find the neighbor with the highest value
        best_neighbor = max(neighbors, key=get_value)
        
        # If no neighbor is strictly better, we reached a peak
        if get_value(best_neighbor) <= get_value(current):
            break
            
        current = best_neighbor
        path.append(current)
        
    return path, current

def run_hill_climbing_multiple_times(graph, num_runs=10):
    nodes = list(graph.keys())
    results = []
    global_optimum_count = 0
    
    for _ in range(num_runs):
        start = random.choice(nodes)
        path, end_state = hill_climbing(graph, start)
        is_global = (end_state == 'Yes')
        if is_global:
            global_optimum_count += 1
        results.append((start, end_state, is_global))
        
    return results, global_optimum_count

# -----------------------------------------------------------------------------
# 2. SIMULATED ANNEALING
# -----------------------------------------------------------------------------
def simulated_annealing(graph, start, initial_temp=100.0, cooling_rate=0.95):
    current = start
    path = [current]
    temp = initial_temp
    
    while temp > 1: # Stop when temperature is very low
        neighbors = graph.get(current, [])
        if not neighbors:
            break
            
        # Pick a random neighbor
        next_node = random.choice(neighbors)
        
        delta_e = get_value(next_node) - get_value(current)
        
        if delta_e > 0:
            current = next_node
            path.append(current)
        else:
            # Probability to accept worse move
            probability = math.exp(delta_e / temp)
            if random.random() < probability:
                current = next_node
                path.append(current)
                
        temp *= cooling_rate
        
    return path, current

# -----------------------------------------------------------------------------
# 3. LOCAL BEAM SEARCH
# -----------------------------------------------------------------------------
def local_beam_search(graph, k=3, max_steps=50):
    nodes = list(graph.keys())
    # Generate k random initial states
    current_states = random.choices(nodes, k=k)
    
    for step in range(max_steps):
        all_neighbors = []
        for state in current_states:
            neighbors = graph.get(state, [])
            if not neighbors:
                all_neighbors.append(state) # keep state if no neighbors
            else:
                all_neighbors.extend(neighbors)
                
        # Select the k best states from all generated neighbors
        best_k_states = sorted(list(set(all_neighbors)), key=get_value, reverse=True)[:k]
        
        # Check if we converged (the states aren't changing)
        if set(best_k_states) == set(current_states):
            break
            
        current_states = best_k_states
        
    return current_states
