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
        
        self.render = render
        self.sim = Simulation(num_agents=1, render=render)

        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float64)

        self.goal = [0, 0, 0]
        self.reward = 0
        self.observation = None
        self.ink = 0

        # Step counter and maximum steps
        self.current_step = 0
        self.max_steps = max_steps

    def reset(self, seed=None):
        # Reset the step counter
        self.current_step = 0

        # Reset other environment states
        self.goal = [random.randint(0, 1), random.randint(0, 1), random.randint(0, 1)]

        state = self.sim.reset(num_agents=1)
        robot_id = next(iter(state))  
        pipette = np.array(state[robot_id]['pipette_position'], dtype=np.float32)

        pipette_delta_x = self.goal[0] - pipette[0]
        pipette_delta_y = self.goal[1] - pipette[1]
        pipette_delta_z = self.goal[2] - pipette[2]

        self.observation = np.concatenate([pipette, [pipette_delta_x, pipette_delta_y, pipette_delta_z]])

        info = {}
        return self.observation, info

    def step(self, action):
        # Increment step counter
        self.current_step += 1

        # Action logic
        velocity_x = random.uniform(-1, 1)
        velocity_y = random.uniform(-1, 1)
        velocity_z = random.uniform(-1, 1)
        drop_command = self.ink

        actions = [[velocity_x, velocity_y, velocity_z, drop_command],
                   [velocity_x, velocity_y, velocity_z, drop_command]]
        
        state = self.sim.run(actions)

        robot_id = next(iter(state))  
        pipette = np.array(state[robot_id]['pipette_position'], dtype=np.float32)

        pipette_delta_x = self.goal[0] - pipette[0]
        pipette_delta_y = self.goal[1] - pipette[1]
        pipette_delta_z = self.goal[2] - pipette[2]

        self.observation = np.concatenate([pipette, [pipette_delta_x, pipette_delta_y, pipette_delta_z]])

        # Calculate reward
        self.reward, terminated = self._calculate_reward(pipette)

        # Check if the maximum number of steps is reached
        if self.current_step >= self.max_steps:
            terminated = bool(self.current_step >= self.max_steps)

        truncated = False

        # Debugging statements
        print(f"Step {self.current_step}/{self.max_steps}, distance_to_goal: {np.linalg.norm(np.array(self.goal) - pipette)}, terminated: {terminated}, truncated: {truncated}")

        return self.observation, self.reward, terminated, truncated, {}

    def _calculate_reward(self, pipette):
        distance_to_goal = np.linalg.norm(np.array(self.goal) - pipette)

        # Base reward for proximity to goal
        reward = -distance_to_goal

        # Reward progress
        if hasattr(self, 'previous_distance_to_goal'):
            progress_reward = self.previous_distance_to_goal - distance_to_goal
            reward += progress_reward * 1.0  # Emphasize progress
        self.previous_distance_to_goal = distance_to_goal

        # Proximity bonus
        if distance_to_goal < 1.5:
            reward += 5 * (1.5 - distance_to_goal)  # Higher reward closer to goal

        # Reduce action penalty
        action_magnitude = np.linalg.norm([random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)])
        reward -= 0.001 * action_magnitude  # Lower penalty
        reward -= 0.005  # Lower time penalty

        # Termination condition
        terminated = bool(distance_to_goal < 1)
        if terminated:
            reward += 10  # Large reward for reaching the goal

        return reward, terminated
