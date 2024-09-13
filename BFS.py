from board import EightPuzzle
from collections import deque
import time

# Define the BFS (Breadth-First Search) class for solving the 8-puzzle problem
class BFS:
    # Initialize the BFS solver with an instance of the EightPuzzle class
    def __init__(self, puzzle):
        self.puzzle = puzzle  # Store the puzzle instance
        self.states_examined = 0  # Initialize the counter for the number of states examined

    # Function to solve the puzzle using the BFS algorithm
    def solve(self):
        visited = set()  # Set to track visited states to avoid revisiting
        queue = deque([(self.puzzle.initial_state, [])])  # Initialize the queue with the initial state and an empty path

        while queue:  # Continue until the queue is empty
            state, path = queue.popleft()  # Dequeue the first state and its associated path
            visited.add(tuple(state.flatten()))  # Mark the state as visited by adding its flattened tuple representation

            self.states_examined += 1  # Increment the count of examined states

            if self.puzzle.is_goal_state(state):  # Check if the current state is the goal state
                return path  # Return the path (sequence of actions) leading to the goal

            actions = self.puzzle.get_possible_actions(state)  # Get the possible actions from the current state
            for action in actions:
                successor_state = self.puzzle.get_successor_state(state, action)  # Generate the successor state for each action
                if tuple(successor_state.flatten()) not in visited:  # Check if the successor state has not been visited
                    queue.append((successor_state, path + [action]))  # Enqueue the successor state and the updated path

        return None  # Return None if no solution is found (i.e., the goal state is not reachable)

# Main execution block
if __name__ == "__main__":
    # Create an instance of the EightPuzzle class with a random initial state
    puzzle = EightPuzzle()

    # Create an instance of the BFS solver with the puzzle instance
    bfs_solver = BFS(puzzle)

    # Measure the start time of the solving process
    start_time = time.time()

    # Solve the puzzle using BFS
    solution = bfs_solver.solve()

    # Measure the end time of the solving process
    end_time = time.time()

    # Calculate the time taken to solve the puzzle
    time_taken = end_time - start_time

    # Output the solution, the number of states examined, and the time taken
    if solution is not None:
        print("Solution found!")
        print("Path:", solution)  # Print the sequence of actions that leads to the goal state
        print("Number of states examined:", bfs_solver.states_examined)  # Print the number of states examined during the search
    else:
        print("No solution found.")
        print("Number of states examined:", bfs_solver.states_examined)  # Print the number of states examined even if no solution is found

    print("Time taken (in seconds):", time_taken)  # Print the total time taken to solve the puzzle