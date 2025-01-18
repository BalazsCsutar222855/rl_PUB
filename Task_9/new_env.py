import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim_class import Simulation
import random

class CustomEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, render=False, max_steps=1000):
        super(CustomEnv, self).__init__()
        
        # Initialize the environment with the render flag
        self.render_flag = render
        self.sim = Simulation(num_agents=1, render=render)

        # Define action space as continuous for X, Y, Z velocity commands
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)  # 3 continuous actions (X, Y, Z)

        # Define observation space: pipette position + delta to goal (6 values)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)

        # Initialize environment state variables
        self.goal = [0, 0, 0]
        self.reward = 0
        self.observation = None
        self.previous_action = None
        self.previous_distance_to_goal = None
        self.step_count = 0
        self.max_steps = max_steps  # Maximum steps per episode

    def reset(self, seed=None):
        # Reset the step count
        self.step_count = 0
        
        # Randomize goal for each episode
        self.goal = [random.randint(-1, 1), random.randint(-1, 1), random.randint(-1, 1)]

        # Reset the simulation state
        state = self.sim.reset(num_agents=1)
        robot_id = next(iter(state))  
        pipette = np.array(state[robot_id]['pipette_position'], dtype=np.float32)

        # Calculate delta between pipette and goal
        pipette_delta_x = self.goal[0] - pipette[0]
        pipette_delta_y = self.goal[1] - pipette[1]
        pipette_delta_z = self.goal[2] - pipette[2]

        # Flatten observation array
        self.observation = np.concatenate([pipette, [pipette_delta_x, pipette_delta_y, pipette_delta_z]])

        # Reset progress tracking
        self.previous_distance_to_goal = None
        self.previous_action = None

        info = {}
        return self.observation, info

    def step(self, action):
        # Action is a continuous 3D velocity command (X, Y, Z)
        velocity_x, velocity_y, velocity_z = action
        
        # Apply action to the simulation (same velocity for both agents)
        actions = [[velocity_x, velocity_y, velocity_z, 0],  # Ink drop can be zero or omitted here
                   [velocity_x, velocity_y, velocity_z, 0]]
        
        state = self.sim.run(actions)

        # Extract pipette position from the simulation state
        robot_id = next(iter(state))  
        pipette = np.array(state[robot_id]['pipette_position'], dtype=np.float32)

        # Calculate delta between pipette and goal
        pipette_delta_x = self.goal[0] - pipette[0]
        pipette_delta_y = self.goal[1] - pipette[1]
        pipette_delta_z = self.goal[2] - pipette[2]

        # Update observation
        self.observation = np.concatenate([pipette, [pipette_delta_x, pipette_delta_y, pipette_delta_z]])

        # Calculate reward based on progress and alignment to goal
        self.reward, terminated = self._calculate_reward(pipette, action)
        truncated = self.step_count >= self.max_steps  # Terminate if max steps exceeded

        self.step_count += 1

        # Info (can include diagnostic information if needed)
        self.info = {}

        return self.observation, self.reward, terminated, truncated, self.info

    def _calculate_reward(self, pipette, action):
        # Calculate distance to goal
        distance_to_goal = np.linalg.norm(np.array(self.goal) - pipette)
        
        # Initialize reward
        reward = 0

        # Reward for progress towards the goal
        if self.previous_distance_to_goal is not None:
            progress = self.previous_distance_to_goal - distance_to_goal
            
            # Penalize if there is very small progress
            if progress < 0.01:
                reward -= 0.1  # Small penalty for insufficient progress
            
            # Reward if significant progress is made
            if progress > 0.1:
                reward += 0.5 * progress  # Scale reward based on progress magnitude

        self.previous_distance_to_goal = distance_to_goal

        # Add a directional reward for aligning movement towards the goal
        direction_to_goal = np.array(self.goal) - pipette
        direction_to_goal /= np.linalg.norm(direction_to_goal)  # Normalize
        if self.previous_action is not None:
            alignment = np.dot(self.previous_action[:3], direction_to_goal)
            reward += 0.1 * alignment  # Small reward for moving in the right direction

        # Add a time penalty to encourage efficiency
        reward -= 0.02  # Time penalty for each step

        # Large reward for reaching the goal
        terminated = distance_to_goal < 0.05
        if terminated:
            reward += 10  # Large reward for goal achievement

        # Save the last action for future alignment reward calculation
        self.previous_action = np.array(action)

        return reward, terminated
