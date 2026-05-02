import pandas as pd
from collections import deque
import copy

import os
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Phase-1'))
df = pd.read_csv(os.path.join(DATA_DIR, 'TrackD_Mental.csv'))
df['work_interfere'] = df['work_interfere'].fillna('Never')
df['self_employed']  = df['self_employed'].fillna('No')

# -------------------------------------------------------
# DEFINE CSP AS DICTIONARY
# -------------------------------------------------------
csp_original = {
    'work_interfere' : ['Never', 'Rarely', 'Sometimes', 'Often'],
    'benefits'       : ['Yes', 'No', "Don't know"],
    'care_options'   : ['Yes', 'No', 'Not sure'],
    'seek_help'      : ['Yes', 'No', "Don't know"],
    'family_history' : ['Yes', 'No']
}
csp = copy.deepcopy(csp_original)



# -------------------------------------------------------
# DEFINE CONSTRAINTS
# Each constraint is a function that takes two values
# and returns True if they are compatible
# -------------------------------------------------------

def constraint_benefits_care(val_benefits, val_care):
    # C4: If care_options = Yes, benefits cannot be No
    if val_care == 'Yes' and val_benefits == 'No':
        return False
    return True

def constraint_seek_benefits(val_seek, val_benefits):
    # C3: If benefits = No, seek_help cannot be Yes
    if val_benefits == 'No' and val_seek == 'Yes':
        return False
    return True

# Map of arcs to their constraint functions
constraints = {
    ('benefits', 'care_options') : constraint_benefits_care,
    ('care_options', 'benefits') : lambda a, b: constraint_benefits_care(b, a),
    ('seek_help', 'benefits')    : constraint_seek_benefits,
    ('benefits', 'seek_help')    : lambda a, b: constraint_seek_benefits(b, a),
}

# List of arcs
arcs = list(constraints.keys())

# -------------------------------------------------------
# REVISE FUNCTION
# Removes values from Xi's domain that have no consistent
# value in Xj's domain
# -------------------------------------------------------

def revise(csp, Xi, Xj, constraints):
    revised = False
    constraint_fn = constraints.get((Xi, Xj))

    if constraint_fn is None:
        return False  # no constraint between these two

    values_to_remove = []
    for val_i in csp[Xi]:
        # Check if there is ANY value in Xj's domain consistent with val_i
        consistent = any(constraint_fn(val_i, val_j) for val_j in csp[Xj])
        if not consistent:
            values_to_remove.append(val_i)

    for val in values_to_remove:
        csp[Xi].remove(val)
        revised = True

    return revised

# -------------------------------------------------------
# AC-3 ALGORITHM
# -------------------------------------------------------

def ac3(csp, arcs, constraints):
    queue = deque(arcs)
    print("\n---- Running AC-3 ----")

    while queue:
        (Xi, Xj) = queue.popleft()
        print(f"  Processing arc: ({Xi}, {Xj})")

        if revise(csp, Xi, Xj, constraints):
            if len(csp[Xi]) == 0:
                print(f" Domain of {Xi} is empty — no solution!")
                return False
            # Re-add all arcs (Xk, Xi) where Xk is a neighbor of Xi
            for (Xk, Xl) in arcs:
                if Xl == Xi and Xk != Xj:
                    queue.append((Xk, Xi))

    return True

if __name__ == '__main__':
    print("----- Domain Sizes BEFORE AC-3 -----")
    for var, domain in csp.items():
        print(f"  {var}: {len(domain)} values: {domain}")

    # Run AC-3
    result = ac3(csp, arcs, constraints)

    print("\n---- Domain Sizes AFTER AC-3 ----")
    for var, domain in csp.items():
        print(f"  {var}: {len(domain)} values: {domain}")

    print("\nAC-3 completed successfully:", result)