import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim_class import Simulation
import random
from collections import deque

class CustomEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, render=False, max_steps=1000):
        super(CustomEnv, self).__init__()
        
        # Now the render argument is accepted
        self.render_flag = render
        self.sim = Simulation(num_agents=1, render=render)

        # Define action space: Discrete 6 actions
        self.action_space = spaces.Discrete(6)
        
        # Define the possible actions (velocities and ink drop command)
        self.actions = [
            [1, 0, 0, 0],  # Action 0: Move in positive x direction
            [0, 1, 0, 0],  # Action 1: Move in positive y direction
            [0, 0, 1, 0],  # Action 2: Move in positive z direction
            [-1, 0, 0, 0], # Action 3: Move in negative x direction
            [0, -1, 0, 0], # Action 4: Move in negative y direction
            [0, 0, -1, 0]  # Action 5: Move in negative z direction
        ]

        # Define observation space: 6 values (pipette position + deltas)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float64)

        # Initialize environment state variables
        self.goal = [0, 0, 0]
        self.reward = 0
        self.observation = None
        self.ink = 0
        self.previous_action = None
        self.previous_distance_to_goal = None
        self.step_count = 0
        self.max_steps = max_steps  # Maximum steps per episode

    def reset(self, seed=None):
        # Reset the step count
        self.step_count = 0
        
        # Reset other environment states
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
        # Select action from the action list
        velocity_x, velocity_y, velocity_z, drop_command = self.actions[action]
        
        actions = [[velocity_x, velocity_y, velocity_z, drop_command],
                   [velocity_x, velocity_y, velocity_z, drop_command]] 
        
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

        # Calculate reward
        self.reward, terminated = self._calculate_reward(pipette, [velocity_x, velocity_y, velocity_z])
        truncated = self.step_count >= self.max_steps  # Truncate if max steps are exceeded

        self.step_count += 1

        # Info (can include diagnostic information if needed)
        self.info = {}

        return self.observation, self.reward, terminated, truncated, self.info
            
    def _drop_ink(self):
        # Ink dropping functionality (if needed)
        self.ink = 1 if self.ink == 0 else 0

    def _calculate_reward(self, pipette, action_velocities):
        # Calculate distance to goal
        distance_to_goal = np.linalg.norm(np.array(self.goal) - pipette)
        
        # Initialize reward
        reward = 0

        # Progress reward
        if self.previous_distance_to_goal is not None:
            progress = self.previous_distance_to_goal - distance_to_goal
            
            # Penalize very small progress
            if progress < 0.01:
                reward -= 0.1  # Small penalty for insufficient progress
            
            # Reward significant progress
            if progress > 0.1:
                reward += 0.5 * progress  # Scale the reward based on progress magnitude

        self.previous_distance_to_goal = distance_to_goal

        # Add a directional reward for aligning movement toward the goal
        direction_to_goal = np.array(self.goal) - pipette
        direction_to_goal = direction_to_goal / np.linalg.norm(direction_to_goal)  # Normalize
        if self.previous_action is not None:
            alignment = np.dot(self.previous_action[:3], direction_to_goal)
            reward += 0.1 * alignment  # Small reward for moving in the right direction

        # Add a time penalty to encourage efficiency
        reward -= 0.02  # Time penalty for each step

        # Check if the task is done (reached the goal)
        terminated = distance_to_goal < 0.05

        # Add a large reward for reaching the goal
        if terminated:
            reward += 10  # Large reward for completing the task

        # Save the last action for directional reward calculation
        self.previous_action = np.array(action_velocities)

        return reward, terminated
