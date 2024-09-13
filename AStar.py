from board import EightPuzzle
import numpy as np
import heapq
import time

# Define the AStar (A*) class for solving the 8-puzzle problem
class AStar:
    def __init__(self, puzzle):
        self.puzzle = puzzle  # Store the puzzle instance
        self.states_examined = 0  # Initialize the counter for the number of states examined

    # Function to solve the puzzle using the A* algorithm
    def solve(self):
        visited = set()  # Set to track visited states to avoid revisiting
        pq = []  # Initialize a priority queue (min-heap) for the A* search
        initial_state = self.puzzle.initial_state  # Get the initial state of the puzzle
        initial_state_tuple = tuple(initial_state.flatten())  # Convert the initial state to a tuple to store in the heap
        # Push the initial state onto the priority queue with its priority (f = g + h), cost (g), state, and path
        heapq.heappush(pq, (0 + self.puzzle.manhattan_distance(initial_state), 0, initial_state_tuple, []))
        
        while pq:  # Continue until the priority queue is empty
            _, g, state_tuple, path = heapq.heappop(pq)  # Pop the state with the lowest f-value from the queue
            # Convert the state tuple back to a NumPy array for processing
            state = np.array(state_tuple).reshape((3, 3))
            visited.add(state_tuple)  # Mark the state as visited by adding its tuple representation

            self.states_examined += 1  # Increment the count of examined states

            if self.puzzle.is_goal_state(state):  # Check if the current state is the goal state
                return path  # Return the path (sequence of actions) leading to the goal

            actions = self.puzzle.get_possible_actions(state)  # Get the possible actions from the current state
            for action in actions:
                successor_state = self.puzzle.get_successor_state(state, action)  # Generate the successor state for each action
                successor_state_tuple = tuple(successor_state.flatten())  # Convert the successor state to a tuple for storage
                if successor_state_tuple not in visited:  # Check if the successor state has not been visited
                    new_g = g + 1  # Increment the cost (g) by 1 for each move
                    h = self.puzzle.manhattan_distance(successor_state)  # Calculate the heuristic (h) using Manhattan distance
                    # Push the successor state onto the priority queue with its priority (f = new_g + h), cost (g), state, and path
                    heapq.heappush(pq, (new_g + h, new_g, successor_state_tuple, path + [action]))

        return None  # Return None if no solution is found (i.e., the goal state is not reachable)

# Main execution block
if __name__ == "__main__":
    # Create an instance of the EightPuzzle class with a random initial state
    puzzle = EightPuzzle()

    # Create an instance of the AStar solver with the puzzle instance
    astar_solver = AStar(puzzle)

    # Measure the start time of the solving process
    start_time = time.time()

    # Solve the puzzle using A*
    solution = astar_solver.solve()

    # Measure the end time of the solving process
    end_time = time.time()

    # Calculate the time taken to solve the puzzle
    time_taken = end_time - start_time

    # Output the solution, the number of states examined, and the time taken
    if solution is not None:
        print("Solution found!")
        print("Path:", solution)  # Print the sequence of actions that leads to the goal state
        print("Number of states examined:", astar_solver.states_examined)  # Print the number of states examined during the search
    else:
        print("No solution found.")
        print("Number of states examined:", astar_solver.states_examined)  # Print the number of states examined even if no solution is found

    print("Time taken (in seconds):", time_taken)  # Print the total time taken to solve the puzzle