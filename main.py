from board import EightPuzzle
from BFS import BFS
from DFS import DFS
from AStar import AStar
from ValueIteration import ValueIteration
from PolicyIteration import PolicyIteration
from QLearning import QLearning
import time

# Define a utility function to solve the puzzle using different solver classes and print the results
def solve_and_print(solver_class, puzzle, solver_name):
    # Instantiate the solver with the given puzzle
    solver = solver_class(puzzle)
    start_time = time.time()  # Record the start time
    solution = solver.solve()  # Solve the puzzle using the solver's solve method
    time_taken = time.time() - start_time  # Calculate the time taken to solve the puzzle

    # Check if a solution was found and print the solution path and its length
    if solution is not None:
        print(f"{solver_name} Solution found!")
        print("Path:", solution)  # Print the sequence of actions leading to the goal state
        print("Path Length:", len(solution))  # Print the length of the solution path (number of moves)
    else:
        print(f"No {solver_name} solution found.")  # Print a message if no solution was found
    
    # If the solver has a 'states_examined' attribute, print the number of states examined
    if hasattr(solver, 'states_examined'):
        print(f"Number of states examined ({solver_name}):", solver.states_examined)
    
    # Print the time taken to solve the puzzle
    print(f"{solver_name} Time taken (in seconds):", time_taken)

# Main execution block
if __name__ == "__main__":
    # Create an instance of the EightPuzzle class with a random initial state
    puzzle = EightPuzzle()

    # Solve the puzzle using different algorithms and print the results
    solve_and_print(BFS, puzzle, "BFS")  # Breadth-First Search
    solve_and_print(DFS, puzzle, "DFS")  # Depth-First Search
    solve_and_print(AStar, puzzle, "A*")  # A* Search
    solve_and_print(ValueIteration, puzzle, "Value Iteration")  # Value Iteration
    solve_and_print(PolicyIteration, puzzle, "Policy Iteration")  # Policy Iteration
    solve_and_print(QLearning, puzzle, "Q-Learning")  # Q-Learning