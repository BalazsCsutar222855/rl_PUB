import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim_class import Simulation
import random
from collections import deque

class CustomEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, render=False, max_steps=100):
        super(CustomEnv, self).__init__()
        
        # Initialize parameters
        self.render = render
        self.sim = Simulation(num_agents=1, render=render)
        self.max_steps = max_steps  # Maximum number of steps per episode
        self.current_step = 0  # Keep track of the current step in the episode
        
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float64)

        self.goal = [0, 0, 0]
        self.reward = 0
        self.observation = None
        self.ink = 0

    def reset(self, seed=None):
        # Reset environment for a new episode
        self.goal = [random.randint(0, 1), random.randint(0, 1), random.randint(0, 1)]
        self.current_step = 0  # Reset the step counter

        # Reset simulation state
        state = self.sim.reset(num_agents=1)
        robot_id = next(iter(state))
        pipette = np.array(state[robot_id]['pipette_position'], dtype=np.float32)

        # Delta between pipette and goal
        pipette_delta_x = self.goal[0] - pipette[0]
        pipette_delta_y = self.goal[1] - pipette[1]
        pipette_delta_z = self.goal[2] - pipette[2]

        self.observation = np.concatenate([pipette, [pipette_delta_x, pipette_delta_y, pipette_delta_z]])

        return self.observation, {}

    def step(self, action):
        # Increment step count
        self.current_step += 1

        # Action (Random velocities for the example)
        velocity_x = random.uniform(-1, 1)
        velocity_y = random.uniform(-1, 1)
        velocity_z = random.uniform(-1, 1)
        drop_command = self.ink

        actions = [[velocity_x, velocity_y, velocity_z, drop_command]]

        state = self.sim.run(actions)

        robot_id = next(iter(state))
        pipette = np.array(state[robot_id]['pipette_position'], dtype=np.float32)

        # Update observation
        pipette_delta_x = self.goal[0] - pipette[0]
        pipette_delta_y = self.goal[1] - pipette[1]
        pipette_delta_z = self.goal[2] - pipette[2]

        self.observation = np.concatenate([pipette, [pipette_delta_x, pipette_delta_y, pipette_delta_z]])

        # Calculate reward
        self.reward, terminated = self._calculate_reward(pipette)

        # If the goal is reached or max_steps is reached, terminate the episode
        if terminated or self.current_step >= self.max_steps:
            truncated = False
            terminated = True
        else:
            truncated = False

        # Render the final state if necessary
        if terminated:
            self.render()

        return self.observation, self.reward, terminated, truncated, {}

    def render(self, mode='human'):
        if self.render:
            self.sim.render(mode)

    def _calculate_reward(self, pipette):
        # Calculate distance to goal
        distance_to_goal = np.linalg.norm(np.array(self.goal) - pipette)
        
        reward = -distance_to_goal  # Linear penalty for distance to the goal
        
        # Reward progress: if the distance decreases, give a small bonus
        if hasattr(self, 'previous_distance_to_goal'):
            progress_reward = self.previous_distance_to_goal - distance_to_goal
            reward += progress_reward * 0.2  # Small bonus for reducing distance
        self.previous_distance_to_goal = distance_to_goal
        
        # Add a small action penalty: slight penalty for large movements
        action_magnitude = np.linalg.norm([random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)])
        reward -= 0.01 * action_magnitude  # Penalize large or unnecessary actions
        
        # Add a time penalty: penalize inefficiency
        reward -= 0.05  # Keep a small time penalty for efficiency

        # Check if the task is done (reached the goal)
        terminated = bool(distance_to_goal < 0.05)

        # Add a big reward for reaching the goal
        if terminated:
            reward += 10  # Large reward for reaching the goal
        
        return reward, terminated
