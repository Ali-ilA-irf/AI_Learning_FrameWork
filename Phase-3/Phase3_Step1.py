import pandas as pd
from collections import deque

import os
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Phase-1'))
df = pd.read_csv(os.path.join(DATA_DIR, 'TrackD_Mental.csv'))
df['work_interfere'] = df['work_interfere'].fillna('Never')
df['self_employed']  = df['self_employed'].fillna('No')

# =============================================
# CSP DEFINITION — Track D (Mental Health)
# =============================================

# VARIABLES: The unknown factors we want to assign values to
# X1 = work_interfere    (how much mental health affects work)
# X2 = benefits          (does employer provide mental health benefits)
# X3 = care_options      (does employee know care options available)
# X4 = seek_help         (does employer provide resources to seek help)
# X5 = family_history    (does person have family history of mental illness)

# DOMAINS: Possible values for each variable
# X1 = work_interfere  ['Never', 'Rarely', 'Sometimes', 'Often']
# X2 = benefits        ['Yes', 'No', "Don't know"]
# X3 = care_options    ['Yes', 'No', 'Not sure']
# X4 = seek_help       ['Yes', 'No', "Don't know"]
# X5 = family_history  ['Yes', 'No']

# CONSTRAINTS (5 rules):
# C1: If work_interfere = 'Often', treatment must be 'Yes'
# C2: If family_history = 'Yes', treatment is likely 'Yes'
# C3: If benefits = 'No' AND seek_help = 'No', care_options cannot be 'Yes'
# C4: If care_options = 'Yes', benefits cannot be 'No'
# C5: If work_interfere = 'Never', obs_consequence is likely 'No'

# GOAL: A complete valid assignment of values to all variables
#       that satisfies all constraints and predicts treatment = 'Yes' or 'No'

if __name__ == '__main__':
    print("---- CSP Definition ----")
    print("Variables : work_interfere, benefits, care_options, seek_help, family_history")
    print("Domains   :")
    print("  work_interfere : ['Never', 'Rarely', 'Sometimes', 'Often']")
    print("  benefits       : ['Yes', 'No', Don't know]")
    print("  care_options   : ['Yes', 'No', 'Not sure']")
    print("  seek_help      : ['Yes', 'No', Don't know]")
    print("  family_history : ['Yes', 'No']")
    print("Constraints: 5 rules defined above")
    print("Goal: Complete assignment satisfying all constraints")