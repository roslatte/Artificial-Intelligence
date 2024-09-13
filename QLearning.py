from board import EightPuzzle
import random
import time
from collections import defaultdict

# Define the QLearning class for solving the 8-puzzle problem using Q-learning
class QLearning:
    def __init__(self, puzzle, discount_factor=0.9, learning_rate=0.1, epsilon=1.0, epsilon_decay=0.999, min_epsilon=0.1, episodes=10000, max_steps=200):
        self.puzzle = puzzle  # Store the puzzle instance
        self.discount_factor = discount_factor  # Discount factor for future rewards
        self.learning_rate = learning_rate  # Learning rate for Q-value updates
        self.epsilon = epsilon  # Epsilon for the epsilon-greedy policy
        self.epsilon_decay = epsilon_decay  # Epsilon decay rate after each episode
        self.min_epsilon = min_epsilon  # Minimum value for epsilon to prevent it from becoming too small
        self.episodes = episodes  # Total number of episodes to run the Q-learning algorithm
        self.max_steps = max_steps  # Maximum number of steps allowed per episode
        # Q-table initialized to 0 for all state-action pairs using a defaultdict
        self.q_table = defaultdict(lambda: defaultdict(float))
        # Track the length of the shortest solution found during the training
        self.shortest_solution_length = float('inf')

    def choose_action(self, state):
        """Epsilon-greedy policy for action selection."""
        if random.random() < self.epsilon:  # With probability epsilon, choose a random action (exploration)
            return random.choice(self.puzzle.get_possible_actions(state))
        else:
            # Get the Q-values for the current state
            q_values = self.q_table[tuple(state.flatten())]
            max_value = max(q_values.values(), default=0)  # Find the maximum Q-value for the current state
            # Select the action(s) with the maximum Q-value
            best_actions = [action for action, value in q_values.items() if value == max_value]
            # If there are multiple best actions, randomly select one; otherwise, pick any valid action
            return random.choice(best_actions) if best_actions else random.choice(self.puzzle.get_possible_actions(state))

    def update_q_value(self, state, action, reward, next_state):
        """Q-value update rule."""
        # Get the current Q-value for the state-action pair
        current_q = self.q_table[tuple(state.flatten())][action]
        # Get the maximum Q-value for the next state
        next_max_q = max(self.q_table[tuple(next_state.flatten())].values(), default=0)
        # Apply the Q-learning update formula
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * next_max_q - current_q)
        # Update the Q-value for the state-action pair in the Q-table
        self.q_table[tuple(state.flatten())][action] = new_q

    def solve(self):
        # Run the Q-learning algorithm for the specified number of episodes
        for episode in range(self.episodes):
            state = self.puzzle.initial_state  # Start each episode with the initial state
            for step in range(self.max_steps):
                action = self.choose_action(state)  # Choose an action using the epsilon-greedy policy
                next_state = self.puzzle.get_successor_state(state, action)  # Get the next state based on the chosen action
                # Calculate the reward with an additional penalty based on Manhattan distance
                reward = self.puzzle.reward(next_state) - self.puzzle.manhattan_distance(next_state)
                self.update_q_value(state, action, reward, next_state)  # Update the Q-value for the state-action pair
                state = next_state  # Move to the next state
                
                if self.puzzle.is_goal_state(state):  # Check if the goal state is reached
                    if step + 1 < self.shortest_solution_length:  # Update the shortest solution length if a shorter one is found
                        self.shortest_solution_length = step + 1
                        print(f"Episode {episode + 1}: New shortest goal reached in {step + 1} steps.")
                    break  # End the episode if the goal state is reached

            # Decay epsilon after each episode to reduce exploration over time
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            
            if (episode + 1) % 1000 == 0:  # Print progress every 1000 episodes
                print(f"Episode {episode + 1}/{self.episodes} completed. Epsilon: {self.epsilon:.4f}")

        # Extract the optimal policy (sequence of actions) based on the learned Q-values
        policy = []
        state = self.puzzle.initial_state  # Start with the initial state
        while not self.puzzle.is_goal_state(state):  # Continue until the goal state is reached
            action = self.choose_action(state)  # Choose the best action based on the Q-table
            policy.append(action)  # Add the action to the policy
            state = self.puzzle.get_successor_state(state, action)  # Move to the next state according to the chosen action

        return policy  # Return the optimal policy (sequence of actions)

# Main execution block
if __name__ == "__main__":
    # Create an instance of the EightPuzzle class with a random initial state
    puzzle = EightPuzzle(initial_state=[[1, 2, 3], [4, 5, 6], [7, 0, 8]])

    # Solve the puzzle using Q-learning
    ql_solver = QLearning(puzzle, episodes=1000, max_steps=10)
    start_time = time.time()  # Measure the start time
    ql_solution = ql_solver.solve()  # Solve the puzzle using the Q-learning algorithm
    end_time = time.time()  # Measure the end time
    ql_time_taken = end_time - start_time  # Calculate the time taken to solve the puzzle

    # Output Q-learning results
    print("\nQ-Learning Solution found!")
    print("Path:", ql_solution)  # Print the sequence of actions that leads to the goal state
    print("Q-Learning Time taken (in seconds):", ql_time_taken)  # Print the total time taken to solve the puzzle