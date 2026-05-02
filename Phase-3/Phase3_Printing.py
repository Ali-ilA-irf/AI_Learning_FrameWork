import copy
import Phase3_Step3 as p3s3
from Phase3_Step2 import csp_original, constraints, arcs, ac3
from Phase3_Step3 import backtracking, forward_checking, backtracking_mrv
from Phase3_Step4 import min_conflicts

print("=" * 60)
print("PHASE 3: CONSTRAINT SATISFACTION PROBLEM RESULTS")
print("=" * 60)

# =============================================================================
# STEP 2: CONSTRAINT PROPAGATION (AC-3)
# =============================================================================
print("\n[STEP 2: CONSTRAINT PROPAGATION & INFERENCE]")
csp_ac3 = copy.deepcopy(csp_original)

print("----- Domain Sizes BEFORE AC-3 -----")
for var, domain in csp_ac3.items():
    print(f"  {var}: {len(domain)} values: {domain}")

print("\n---- Running AC-3 ----")
result = ac3(csp_ac3, arcs, constraints)

print("\n---- Domain Sizes AFTER AC-3 ----")
for var, domain in csp_ac3.items():
    print(f"  {var}: {len(domain)} values: {domain}")
print(f"AC-3 completed successfully: {result}")

# =============================================================================
# STEP 3: BACKTRACKING SEARCH
# =============================================================================
print("\n[STEP 3: BACKTRACKING SEARCH]")

# 1. Plain Backtracking
csp_bt = copy.deepcopy(csp_original)
p3s3.backtrack_count = 0
solution_bt = backtracking({}, csp_bt, constraints)
print("\n----- Plain Backtracking -----")
print("Solution:", solution_bt)
print("Backtracks:", p3s3.backtrack_count)

# 2. Forward Checking
csp_fc = copy.deepcopy(csp_original)
p3s3.fc_backtrack_count = 0
solution_fc = forward_checking({}, csp_fc, constraints)
print("\n----- Backtracking with Forward Checking -----")
print("Solution:", solution_fc)
print("Backtracks:", p3s3.fc_backtrack_count)

# 3. MRV Heuristic
csp_mrv = copy.deepcopy(csp_original)
p3s3.mrv_backtrack_count = 0
solution_mrv = backtracking_mrv({}, csp_mrv, constraints)
print("\n----- Backtracking with MRV -----")
print("Solution:", solution_mrv)
print("Backtracks:", p3s3.mrv_backtrack_count)


# =============================================================================
# STEP 4: LOCAL SEARCH FOR CSP
# =============================================================================
print("\n[STEP 4: LOCAL SEARCH FOR CSP]")

print("----- Min-Conflicts -----")
solution_mc, steps_mc = min_conflicts(csp_original, constraints, max_steps=100)
print("\nFinal Assignment:", solution_mc)
print("Steps Taken:", steps_mc)

print("=" * 60)
