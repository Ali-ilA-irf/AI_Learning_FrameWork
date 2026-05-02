from Phase2_Step1 import graph
from Phase2_Step4 import get_value

# -----------------------------------------------------------------------------
# 1. MINIMAX SEARCH
# -----------------------------------------------------------------------------
def minimax(graph, current_state, depth, is_maximizing, counter=None):
    if counter is not None:
        counter[0] += 1
        
    if depth == 0 or current_state == 'Yes':
        return get_value(current_state), None
        
    neighbors = graph.get(current_state, [])
    if not neighbors:
        return get_value(current_state), None

    if is_maximizing:
        best_val = -float('inf')
        best_move = None
        for neighbor in neighbors:
            val, _ = minimax(graph, neighbor, depth - 1, False, counter)
            if val > best_val:
                best_val = val
                best_move = neighbor
        if depth < 4:  # Don't print for the leaf level, print the chosen move at the current depth
             print(f"    [Minimax] Depth {depth} (MAX): Chose move '{best_move}' with value {best_val}")
        return best_val, best_move
    else:
        best_val = float('inf')
        best_move = None
        for neighbor in neighbors:
            val, _ = minimax(graph, neighbor, depth - 1, True, counter)
            if val < best_val:
                best_val = val
                best_move = neighbor
        if depth < 4:
             print(f"    [Minimax] Depth {depth} (MIN): Chose move '{best_move}' with value {best_val}")
        return best_val, best_move

# -----------------------------------------------------------------------------
# 2. ALPHA-BETA PRUNING
# -----------------------------------------------------------------------------
def alphabeta(graph, current_state, depth, alpha, beta, is_maximizing, counter=None):
    if counter is not None:
        counter[0] += 1
        
    if depth == 0 or current_state == 'Yes':
        return get_value(current_state), None
        
    neighbors = graph.get(current_state, [])
    if not neighbors:
        return get_value(current_state), None

    if is_maximizing:
        best_val = -float('inf')
        best_move = None
        for neighbor in neighbors:
            val, _ = alphabeta(graph, neighbor, depth - 1, alpha, beta, False, counter)
            if val > best_val:
                best_val = val
                best_move = neighbor
            alpha = max(alpha, best_val)
            if beta <= alpha:
                break # Beta pruning
        return best_val, best_move
    else:
        best_val = float('inf')
        best_move = None
        for neighbor in neighbors:
            val, _ = alphabeta(graph, neighbor, depth - 1, alpha, beta, True, counter)
            if val < best_val:
                best_val = val
                best_move = neighbor
            beta = min(beta, best_val)
            if beta <= alpha:
                break # Alpha pruning
        return best_val, best_move
