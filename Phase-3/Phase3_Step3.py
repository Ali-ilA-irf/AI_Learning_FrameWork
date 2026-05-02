import copy
from Phase3_Step2 import csp_original, constraints, ac3, arcs

def is_consistent(assignment, var, value, constraints):
    for (Xi, Xj), fn in constraints.items():
        if Xi == var and Xj in assignment:
            if not fn(value, assignment[Xj]):
                return False
        if Xj == var and Xi in assignment:
            if not fn(assignment[Xi], value):
                return False
    return True


# =============================================
# PLAIN BACKTRACKING
# =============================================
backtrack_count = 0

def backtracking(assignment, csp, constraints):
    global backtrack_count

    if len(assignment) == len(csp):
        return assignment

    unassigned = [v for v in csp if v not in assignment]
    var = unassigned[0]

    for value in csp[var]:
        if is_consistent(assignment, var, value, constraints):
            assignment[var] = value
            result = backtracking(assignment, csp, constraints)
            if result is not None:
                return result
            del assignment[var]
            backtrack_count += 1

    return None

# =============================================
# BACKTRACKING WITH FORWARD CHECKING
# =============================================
fc_backtrack_count = 0

def forward_checking(assignment, csp, constraints):
    global fc_backtrack_count

    if len(assignment) == len(csp):
        return assignment

    unassigned = [v for v in csp if v not in assignment]
    var = unassigned[0]

    for value in csp[var]:
        if is_consistent(assignment, var, value, constraints):
            assignment[var] = value
            
            # Forward Checking
            local_csp = copy.deepcopy(csp)
            domain_wipeout = False
            
            # Only checking simple constraints between assigned `var` and unassigned neighbors
            for un_var in unassigned[1:]:
                values_to_keep = []
                for un_val in local_csp[un_var]:
                    temp_assign = copy.deepcopy(assignment)
                    temp_assign[un_var] = un_val
                    if is_consistent(temp_assign, un_var, un_val, constraints):
                        values_to_keep.append(un_val)
                
                local_csp[un_var] = values_to_keep
                if len(values_to_keep) == 0:
                    domain_wipeout = True
                    break # Domain empty, stop checking
            
            if not domain_wipeout:
                result = forward_checking(assignment, local_csp, constraints)
                if result is not None:
                    return result
                    
            del assignment[var]
            fc_backtrack_count += 1

    return None

# =============================================
# BACKTRACKING WITH MRV HEURISTIC
# =============================================
mrv_backtrack_count = 0

def backtracking_mrv(assignment, csp, constraints):
    global mrv_backtrack_count

    if len(assignment) == len(csp):
        return assignment

    unassigned = [v for v in csp if v not in assignment]
    # MRV: Pick variable with fewest remaining values in domain
    var = min(unassigned, key=lambda v: len(csp[v]))

    for value in csp[var]:
        if is_consistent(assignment, var, value, constraints):
            assignment[var] = value
            result = backtracking_mrv(assignment, csp, constraints)
            if result is not None:
                return result
            del assignment[var]
            mrv_backtrack_count += 1

    return None

if __name__ == '__main__':
    print("---  STEP 3: BACKTRACKING ---")

    # Plain BT
    csp_bt = copy.deepcopy(csp_original)
    backtrack_count = 0
    solution_bt = backtracking({}, csp_bt, constraints)
    print("\n----- Plain Backtracking -----")
    print("Solution :", solution_bt)
    print("Backtracks:", backtrack_count)

    # Forward Checking
    csp_fc = copy.deepcopy(csp_original)
    fc_backtrack_count = 0
    solution_fc = forward_checking({}, csp_fc, constraints)
    print("\n----- Backtracking with Forward Checking -----")
    print("Solution :", solution_fc)
    print("Backtracks:", fc_backtrack_count)

    # MRV
    csp_mrv = copy.deepcopy(csp_original)
    mrv_backtrack_count = 0
    solution_mrv = backtracking_mrv({}, csp_mrv, constraints)
    print("\n----- Backtracking with MRV -----")
    print("Solution :", solution_mrv)
    print("Backtracks:", mrv_backtrack_count)

