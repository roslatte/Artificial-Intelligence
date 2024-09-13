from board import EightPuzzle
import numpy as np
import time
import itertools

# Define the ValueIteration class for solving the 8-puzzle problem using the value iteration algorithm
class ValueIteration:
    def __init__(self, puzzle, discount_factor=0.75, theta=1e-2):
        self.puzzle = puzzle  # Store the puzzle instance
        self.discount_factor = discount_factor  # Discount factor for future rewards
        self.theta = theta  # Convergence threshold
        self.V = {}  # Dictionary to store the value function for each state

        # Initialize the value function for all possible states
        self.initialize_value_function()

    def initialize_value_function(self):
        # Generate all possible states of the 8-puzzle using permutations of 0-8
        all_possible_states = itertools.permutations(range(9))
        for state_tuple in all_possible_states:
            state_array = np.array(state_tuple).reshape(3, 3)  # Convert the tuple to a 3x3 array
            if self.puzzle.is_goal_state(state_array):  # Check if the state is the goal state
                self.V[state_tuple] = 0  # Goal state has a value of 0
            else:
                self.V[state_tuple] = -self.puzzle.manhattan_distance(state_array)  # Initialize value based on Manhattan distance

    def solve(self):
        iteration = 0  # Initialize iteration counter
        while True:
            delta = 0  # Initialize delta to track the maximum change in value function
            iteration += 1  # Increment iteration counter

            # Iterate over all states in the value function
            for state_tuple in list(self.V.keys()):
                state = np.array(state_tuple).reshape(3, 3)  # Convert tuple back to 3x3 array
                if self.puzzle.is_goal_state(state):  # Skip goal state since its value is already 0
                    continue

                v = self.V[state_tuple]  # Current value of the state
                max_value = float('-inf')  # Initialize max value for possible actions
                actions = self.puzzle.get_possible_actions(state)  # Get possible actions from the current state

                # Evaluate all possible actions and update the value function
                for action in actions:
                    successor_state = self.puzzle.get_successor_state(state, action)  # Get the successor state for each action
                    successor_state_tuple = tuple(successor_state.flatten())  # Convert the successor state to a tuple
                    reward = self.puzzle.reward(successor_state)  # Get the reward for the successor state
                    value = reward + self.discount_factor * self.V[successor_state_tuple]  # Calculate the value based on the Bellman equation
                    max_value = max(max_value, value)  # Update max value for the best action

                self.V[state_tuple] = max_value  # Update the value function with the max value
                delta = max(delta, abs(v - self.V[state_tuple]))  # Update delta with the largest change in value function

            print(f"Iteration: {iteration}, Delta : {delta:.6f}")  # Print iteration and delta for tracking convergence

            # Check for convergence
            if delta < self.theta:  # If delta is less than the threshold, consider the algorithm converged
                print("Convergence achieved.")
                break

        # Extract the optimal policy (sequence of actions) from the value function
        policy = []
        state = self.puzzle.initial_state  # Start with the initial state
        while not self.puzzle.is_goal_state(state):  # Continue until the goal state is reached
            actions = self.puzzle.get_possible_actions(state)  # Get possible actions from the current state
            best_action = None
            max_value = float('-inf')
            for action in actions:
                successor_state = self.puzzle.get_successor_state(state, action)  # Get the successor state for each action
                successor_state_tuple = tuple(successor_state.flatten())  # Convert the successor state to a tuple
                value = self.V[successor_state_tuple]  # Get the value of the successor state from the value function
                if value > max_value:  # Select the action that maximizes the value function
                    max_value = value
                    best_action = action
            policy.append(best_action)  # Add the best action to the policy
            state = self.puzzle.get_successor_state(state, best_action)  # Move to the next state according to the best action

        return policy  # Return the optimal policy (sequence of actions)

# Main execution block
if __name__ == "__main__":
    # Create an instance of the EightPuzzle class with a set initial state for speed of running
    puzzle = EightPuzzle(initial_state=[[1, 2, 3], [4, 5, 6], [7, 0, 8]])

    # Solve the puzzle using Value Iteration
    vi_solver = ValueIteration(puzzle, discount_factor = 0, theta = 0.1)
    start_time = time.time()  # Measure the start time
    vi_solution = vi_solver.solve()  # Solve the puzzle using the value iteration algorithm
    end_time = time.time()  # Measure the end time
    vi_time_taken = end_time - start_time  # Calculate the time taken to solve the puzzle

    # Output Value Iteration results
    print("\nValue Iteration Solution found!")
    print("Path:", vi_solution)  # Print the sequence of actions that leads to the goal state
    print("Value Iteration Time taken (in seconds):", vi_time_taken)  # Print the total time taken to solve the puzzle