import numpy as np
from copy import deepcopy
import random

# Define the EightPuzzle class which represents the 8-puzzle problem
class EightPuzzle:
    # Initialize the puzzle with an initial state (either provided or generated randomly)
    def __init__(self, initial_state=None):
        if initial_state is None:
            self.initial_state = self.generate_random_state()  # Generate a random state if none is provided
        else:
            self.initial_state = np.array(initial_state)  # Convert the provided state to a NumPy array
        self.goal_state = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 0]])  # Define the goal state

    # Helper function to generate a random, solvable initial state
    def generate_random_state(self):
        while True:
            numbers = list(range(9))  # Create a list of numbers from 0 to 8
            random.shuffle(numbers)  # Shuffle the list to generate a random order
            state = np.array([numbers[i:i+3] for i in range(0, 9, 3)])  # Convert the list into a 3x3 NumPy array
            if self.is_solvable(state):  # Check if the generated state is solvable
                return state  # Return the solvable state

    # Helper function to check if a given state is solvable
    def is_solvable(self, state):
        flattened = state.flatten()  # Flatten the 2D state array into a 1D array
        flattened = flattened[flattened != 0]  # Remove the zero (blank space) from the array
        inversions = 0  # Initialize the inversion count
        for i in range(len(flattened)):
            for j in range(i + 1, len(flattened)):
                if flattened[i] > flattened[j]:  # Count the number of inversions
                    inversions += 1
        return inversions % 2 == 0  # A state is solvable if the number of inversions is even

    # Helper function to check if the current state is the goal state
    def is_goal_state(self, state):
        return np.array_equal(state, self.goal_state)  # Return True if the current state matches the goal state

    # Helper function to get the possible actions ('up', 'down', 'left', 'right') for the blank space
    def get_possible_actions(self, state):
        actions = []  # Initialize an empty list for actions
        zero_pos = np.argwhere(state == 0)[0]  # Find the position of the blank space (zero)
        row, col = zero_pos[0], zero_pos[1]  # Extract the row and column indices of the blank space

        if row > 0:
            actions.append('up')  # The blank space can move up if it's not in the top row
        if row < 2:
            actions.append('down')  # The blank space can move down if it's not in the bottom row
        if col > 0:
            actions.append('left')  # The blank space can move left if it's not in the leftmost column
        if col < 2:
            actions.append('right')  # The blank space can move right if it's not in the rightmost column

        return actions  # Return the list of possible actions

    # Helper function to get the successor state after taking a specified action
    def get_successor_state(self, state, action):
        successor_state = deepcopy(state)  # Create a deep copy of the current state
        zero_pos = np.argwhere(successor_state == 0)[0]  # Find the position of the blank space
        row, col = zero_pos[0], zero_pos[1]  # Extract the row and column indices of the blank space

        # Swap the blank space with the appropriate adjacent tile based on the action
        if action == 'up':
            successor_state[row, col], successor_state[row - 1, col] = successor_state[row - 1, col], successor_state[row, col]
        elif action == 'down':
            successor_state[row, col], successor_state[row + 1, col] = successor_state[row + 1, col], successor_state[row, col]
        elif action == 'left':
            successor_state[row, col], successor_state[row, col - 1] = successor_state[row, col - 1], successor_state[row, col]
        elif action == 'right':
            successor_state[row, col], successor_state[row, col + 1] = successor_state[row, col + 1], successor_state[row, col]

        return successor_state  # Return the new state after applying the action

    # Helper function to get the reward for a given state
    def reward(self, state):
        if self.is_goal_state(state):
            return 0  # No cost if the goal is reached
        else:
            return -1  # Cost of each move if the goal is not reached

    # Helper function to calculate the Manhattan distance heuristic for the given state
    def manhattan_distance(self, state):
        distance = 0  # Initialize the Manhattan distance
        for i in range(3):
            for j in range(3):
                if state[i, j] != 0:  # Skip the blank space (zero)
                    goal_pos = np.argwhere(self.goal_state == state[i, j])[0]  # Find the position of the tile in the goal state
                    # Add the Manhattan distance between the current position and the goal position of the tile
                    distance += abs(goal_pos[0] - i) + abs(goal_pos[1] - j)
        return distance  # Return the total Manhattan distance for the state