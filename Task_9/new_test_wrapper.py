import gymnasium as gym
import numpy as np
from new_ot2_gym_wrapper import OT2Env  # Make sure to import your environment correctly


# Load your custom environment
env = OT2Env(render=False)

# Number of episodes
num_episodes = 5

# Initialize variables to track the overall minimum distance to goal and highest reward
overall_min_distance_to_goal = float('inf')  # Start with a large value
overall_max_reward = float('-inf')  # Start with a very low reward

for episode in range(num_episodes):
    obs = env.reset()
    done = False
    step = 0

    # Initialize the minimum distance to the goal for this episode
    min_distance_to_goal = np.linalg.norm(obs[:3] - env.goal_position)
    max_reward = float('-inf')  # Track max reward for this episode

    while not done:
        # Take a random action from the environment's action space
        action = env.action_space.sample()

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)

        # Calculate the current distance to the goal
        distance_to_goal = np.linalg.norm(obs[:3] - env.goal_position)

        # Update the minimum distance to the goal for this episode
        if distance_to_goal < min_distance_to_goal:
            min_distance_to_goal = distance_to_goal

        # Update the max reward for this episode
        if reward > max_reward:
            max_reward = reward
        
        step += 1

        # Check if the episode is finished
        if terminated or truncated:
            print(f"Episode {episode + 1} finished after {step} steps.")
            print(f"  - Lowest Distance to Goal this episode: {min_distance_to_goal:.4f}")
            print(f"  - Highest Reward this episode: {max_reward}")
            break

    # Track the overall minimum distance and highest reward
    if min_distance_to_goal < overall_min_distance_to_goal:
        overall_min_distance_to_goal = min_distance_to_goal
    if max_reward > overall_max_reward:
        overall_max_reward = max_reward

# After all episodes, log the overall best results
print(f"\nOverall Results after {num_episodes} episodes:")
print(f"  - Smallest Distance to Goal: {overall_min_distance_to_goal:.4f}")
print(f"  - Highest Reward: {overall_max_reward}")