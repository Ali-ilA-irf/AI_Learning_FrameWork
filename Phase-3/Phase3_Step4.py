import random
import copy
from Phase3_Step2 import csp_original, constraints
from Phase3_Step3 import is_consistent

# =============================================
# MIN-CONFLICTS LOCAL SEARCH
# =============================================

def count_conflicts(assignment, var, value, constraints):
    # Temporarily assign value to var to check conflicts
    temp_assign = copy.deepcopy(assignment)
    temp_assign[var] = value
    conflicts = 0
    
    for (Xi, Xj), fn in constraints.items():
        if Xi in temp_assign and Xj in temp_assign:
            if not fn(temp_assign[Xi], temp_assign[Xj]):
                conflicts += 1
    return conflicts

def min_conflicts(csp, constraints, max_steps=100):
    # 1. Start with a random complete assignment
    current_assignment = {}
    for var in csp:
        current_assignment[var] = random.choice(csp[var])
        
    print(f"Initial Random Assignment: {current_assignment}")
    
    for step in range(max_steps):
        # 2. Find violated constraints
        violated_vars = set()
        for (Xi, Xj), fn in constraints.items():
            if not fn(current_assignment[Xi], current_assignment[Xj]):
                violated_vars.add(Xi)
                violated_vars.add(Xj)
                
        # If no violations, we found a solution!
        if not violated_vars:
            print(f"Solution found at step {step}!")
            return current_assignment, step
            
        # 3. Select a random variable involved in a conflict
        var_to_change = random.choice(list(violated_vars))
        
        # 4. Assign the value that minimizes conflicts
        min_conflict_val = current_assignment[var_to_change]
        min_conflict_count = float('inf')
        
        for val in csp[var_to_change]:
            conflicts = count_conflicts(current_assignment, var_to_change, val, constraints)
            if conflicts < min_conflict_count:
                min_conflict_count = conflicts
                min_conflict_val = val
                
        current_assignment[var_to_change] = min_conflict_val
        
    print(f"Failed to find solution within {max_steps} steps.")
    return current_assignment, max_steps

if __name__ == '__main__':
    print("--- STEP 4: MIN-CONFLICTS LOCAL SEARCH ---")
    solution_mc, steps_mc = min_conflicts(csp_original, constraints, max_steps=100)
    print("\n----- Min-Conflicts Result -----")
    print("Final Assignment:", solution_mc)
    print("Steps Taken:", steps_mc)
