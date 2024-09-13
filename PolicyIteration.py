from board import EightPuzzle
import numpy as np
import itertools
import random
import time

# Define the PolicyIteration class for solving the 8-puzzle problem using policy iteration
class PolicyIteration:
    def __init__(self, puzzle, discount_factor=0.75, theta=1e-2):
        self.puzzle = puzzle  # Store the puzzle instance
        self.discount_factor = discount_factor  # Discount factor for future rewards
        self.theta = theta  # Convergence threshold
        self.V = {}  # Dictionary to store the value of each state
        self.policy = {}  # Dictionary to store the policy for each state
        self.successor_cache = {}  # Cache for successor states to avoid redundant calculations

        # Initialize the value function and policy for all possible states
        self.initialize_value_function_and_policy()

    def initialize_value_function_and_policy(self):
        # Generate all possible states (permutations of tiles 0-8)
        all_possible_states = itertools.permutations(range(9))
        for state_tuple in all_possible_states:
            state_array = np.array(state_tuple).reshape(3, 3)  # Convert tuple to 3x3 array
            self.V[state_tuple] = -self.puzzle.manhattan_distance(state_array)  # Initialize value based on Manhattan distance
            # Initialize policy with a random valid action for each state
            self.policy[state_tuple] = random.choice(self.puzzle.get_possible_actions(state_array))

    def get_cached_successor(self, state, action):
        # Cache the successor states to avoid redundant calculations
        state_tuple = tuple(state.flatten())
        if (state_tuple, action) not in self.successor_cache:
            # If the successor state is not in the cache, compute and store it
            self.successor_cache[(state_tuple, action)] = self.puzzle.get_successor_state(state, action)
        return self.successor_cache[(state_tuple, action)]

    def policy_evaluation(self):
        # Evaluate the current policy by iteratively updating the value function
        while True:
            delta = 0  # Initialize delta to track the maximum change in value function
            for state_tuple in list(self.V.keys()):
                state = np.array(state_tuple).reshape(3, 3)  # Convert tuple back to 3x3 array
                if self.puzzle.is_goal_state(state):  # Skip goal state since its value is already 0
                    continue

                v = self.V[state_tuple]  # Current value of the state
                action = self.policy[state_tuple]  # Get the action dictated by the current policy
                successor_state = self.get_cached_successor(state, action)  # Get the successor state based on the action
                successor_state_tuple = tuple(successor_state.flatten())  # Convert successor state to tuple
                reward = self.puzzle.reward(successor_state)  # Get the reward for the successor state
                # Update the value function using the Bellman equation
                self.V[state_tuple] = reward + self.discount_factor * self.V[successor_state_tuple]
                delta = max(delta, abs(v - self.V[state_tuple]))  # Update delta with the largest change in value

            if delta < self.theta:  # If delta is less than the threshold, consider the policy evaluation converged
                break

        return delta  # Return delta for reporting

    def policy_improvement(self):
        # Improve the current policy by choosing the best action for each state
        policy_stable = True  # Initialize a flag to check if the policy is stable
        for state_tuple in list(self.V.keys()):
            state = np.array(state_tuple).reshape(3, 3)  # Convert tuple back to 3x3 array
            if self.puzzle.is_goal_state(state):  # Skip goal state since no action is needed
                continue

            old_action = self.policy[state_tuple]  # Store the current action for comparison
            actions = self.puzzle.get_possible_actions(state)  # Get possible actions from the current state

            best_action = None  # Initialize best action and max value
            max_value = float('-inf')
            # Evaluate each action and choose the one with the maximum value
            for action in actions:
                successor_state = self.get_cached_successor(state, action)  # Get the successor state for each action
                successor_state_tuple = tuple(successor_state.flatten())  # Convert successor state to tuple
                value = self.puzzle.reward(successor_state) + self.discount_factor * self.V[successor_state_tuple]  # Compute value
                if value > max_value:  # Update the best action if this value is the highest
                    max_value = value
                    best_action = action

            self.policy[state_tuple] = best_action  # Update the policy with the best action

            if old_action != best_action:  # If the action changed, the policy is not stable
                policy_stable = False

        return policy_stable  # Return whether the policy is stable

    def solve(self):
        iteration = 0  # Initialize iteration counter
        while True:
            iteration += 1  # Increment iteration counter
            delta = self.policy_evaluation()  # Evaluate the current policy
            print(f"Policy Iteration: {iteration}, Delta: {delta:.6f}")  # Print iteration and delta for tracking

            if self.policy_improvement():  # Improve the policy and check if it's stable
                print("Policy Converged.")  # If the policy is stable, convergence is achieved
                break

        # Extract the optimal policy (sequence of actions) from the policy dictionary
        optimal_policy = []
        state = self.puzzle.initial_state  # Start with the initial state
        while not self.puzzle.is_goal_state(state):  # Continue until the goal state is reached
            state_tuple = tuple(state.flatten())  # Convert state to tuple
            action = self.policy[state_tuple]  # Get the action dictated by the optimal policy
            optimal_policy.append(action)  # Add the action to the policy
            state = self.get_cached_successor(state, action)  # Move to the next state according to the action

        return optimal_policy  # Return the optimal policy (sequence of actions)

# Main execution block
if __name__ == "__main__":
    # Create an instance of the EightPuzzle class with a random initial state
    puzzle = EightPuzzle(initial_state=[[1, 2, 3], [4, 5, 6], [7, 0, 8]])

    # Solve the puzzle using Policy Iteration
    pi_solver = PolicyIteration(puzzle, discount_factor = 0, theta = 0.1)
    start_time = time.time()  # Measure the start time
    pi_solution = pi_solver.solve()  # Solve the puzzle using the policy iteration algorithm
    end_time = time.time()  # Measure the end time
    pi_time_taken = end_time - start_time  # Calculate the time taken to solve the puzzle

    # Output Policy Iteration results
    print("\nPolicy Iteration Solution found!")
    print("Path:", pi_solution)  # Print the sequence of actions that leads to the goal state
    print("Policy Iteration Time taken (in seconds):", pi_time_taken)  # Print the total time taken to solve the puzzle