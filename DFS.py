from board import EightPuzzle
import time

# Define the DFS (Depth-First Search) class for solving the 8-puzzle problem
class DFS:
    def __init__(self, puzzle, max_depth=50):
        self.puzzle = puzzle  # Store the puzzle instance
        self.max_depth = max_depth  # Set the maximum search depth to limit the depth-first search
        self.states_examined = 0  # Initialize the counter for the number of states examined

    # Function to solve the puzzle using the DFS algorithm
    def solve(self):
        visited = set()  # Set to track visited states to avoid revisiting
        stack = [(self.puzzle.initial_state, [], 0)]  # Initialize the stack with the initial state, an empty path, and depth 0

        while stack:  # Continue until the stack is empty
            state, path, depth = stack.pop()  # Pop the last state, its associated path, and the current depth from the stack
            visited.add(tuple(state.flatten()))  # Mark the state as visited by adding its flattened tuple representation

            self.states_examined += 1  # Increment the count of examined states

            if self.puzzle.is_goal_state(state):  # Check if the current state is the goal state
                return path  # Return the path (sequence of actions) leading to the goal

            if depth < self.max_depth:  # Continue searching only if the current depth is less than the maximum depth
                actions = self.puzzle.get_possible_actions(state)  # Get the possible actions from the current state
                for action in actions:
                    successor_state = self.puzzle.get_successor_state(state, action)  # Generate the successor state for each action
                    if tuple(successor_state.flatten()) not in visited:  # Check if the successor state has not been visited
                        stack.append((successor_state, path + [action], depth + 1))  # Push the successor state, updated path, and incremented depth onto the stack

        return None  # Return None if no solution is found (i.e., the goal state is not reachable within the depth limit)

# Main execution block
if __name__ == "__main__":
    # Create an instance of the EightPuzzle class with a random initial state
    puzzle = EightPuzzle()

    # Create an instance of the DFS solver with the puzzle instance
    dfs_solver = DFS(puzzle)

    # Measure the start time of the solving process
    start_time = time.time()

    # Solve the puzzle using DFS
    solution = dfs_solver.solve()

    # Measure the end time of the solving process
    end_time = time.time()

    # Calculate the time taken to solve the puzzle
    time_taken = end_time - start_time

    # Output the solution, the number of states examined, and the time taken
    if solution is not None:
        print("Solution found!")
        print("Path:", solution)  # Print the sequence of actions that leads to the goal state
        print("Number of states examined:", dfs_solver.states_examined)  # Print the number of states examined during the search
    else:
        print("No solution found.")
        print("Number of states examined:", dfs_solver.states_examined)  # Print the number of states examined even if no solution is found

    print("Time taken (in seconds):", time_taken)  # Print the total time taken to solve the puzzle
