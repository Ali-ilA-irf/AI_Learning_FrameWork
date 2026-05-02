import sys
import os
from collections import deque
import heapq

# Add Phase-1 directory to sys.path to allow importing its modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Phase-1')))

# Import Phase-1 Step 6 to get the shared graph
from Phase1_Step6 import graph

# -----------------------------------------------------------------------------
# State space explanation
# Initial State  : A set of observed survey answers (feature values)
# Goal State     : The correct treatment label ('Yes' or 'No')
# Actions        : Moving from one feature-value combination to another
# Cost           : 1 per step (uniform)
# Solution       : The path from initial feature observations to correct label
# ------------------------------------------------------------------------------

class AIAgent:
    def __init__(self, graph, goal):
        self.graph = graph   # the state space graph
        self.goal = goal     # goal state (e.g., 'Yes' or 'No')
        self.current_state = None

    def perceive(self, state):
        """Returns list of possible next states from current state."""
        return self.graph.get(state, [])

    def act(self, action):
        """Moves agent to a new state."""
        self.current_state = action
        print(f"Agent moved to state: {action}")

    def goal_test(self, state):
        """Returns True if the state is the goal."""
        return state == self.goal

    def get_cost(self, state1, state2):
        """Cost of moving between states — uniform cost = 1."""
        return 1

# Create the agent using the imported graph
agent = AIAgent(graph=graph, goal='Yes')

if __name__ == '__main__':
    # The graph is now imported from Phase-1 Step 6
    print("\nGraph imported from Phase-1 Step 6 successfully!")
    print(f"Total nodes: {len(graph)}")

    print("\nAIAgent created using Phase-1 graph.")
    print("Goal state:", agent.goal)


