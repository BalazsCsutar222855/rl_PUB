import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim_class import Simulation
import random


class CustomEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, render=False):
        super(CustomEnv, self).__init__()

        self.render_flag = render
        self.sim = Simulation(num_agents=1, render=render)

        # Discrete action space: map to velocities and ink drop
        self.action_space = spaces.Discrete(6)  # 6 discrete actions

        # Observation space: pipette position and deltas to goal
        self.observation_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)

        # State variables
        self.goal = [0.0, 0.0, 0.0]  # Goal position
        self.reward = 0
        self.observation = None
        self.ink = 0  # Ink drop state

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset the goal to a random position in [0, 1]^3
        self.goal = [random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)]

        # Reset the simulation and get initial pipette position
        state = self.sim.reset(num_agents=1)
        robot_id = next(iter(state))  
        pipette = np.array(state[robot_id]["pipette_position"], dtype=np.float32)

        # Compute deltas to goal
        pipette_deltas = np.array(self.goal) - pipette

        # Initial observation
        self.observation = np.concatenate([pipette, pipette_deltas], axis=0)

        # Reset internal tracking
        self.previous_distance_to_goal = np.linalg.norm(pipette_deltas)

        return self.observation, {}

    def step(self, action):
        # Define discrete actions: velocities and ink drop
        action_map = {
            0: [1, 0, 0],   # Move +x
            1: [-1, 0, 0],  # Move -x
            2: [0, 1, 0],   # Move +y
            3: [0, -1, 0],  # Move -y
            4: [0, 0, 1],   # Move +z
            5: [0, 0, -1],  # Move -z
        }

        velocity = np.array(action_map[action]) * 0.1  # Scale velocity for smoother movement
        drop_command = self.ink

        # Apply the action to the simulation
        actions = [[*velocity, drop_command], [*velocity, drop_command]]
        state = self.sim.run(actions)

        # Extract new pipette position
        robot_id = next(iter(state))  
        pipette = np.array(state[robot_id]["pipette_position"], dtype=np.float32)

        # Compute deltas to goal
        pipette_deltas = np.array(self.goal) - pipette

        # Update observation
        self.observation = np.concatenate([pipette, pipette_deltas], axis=0)

        # Calculate reward and termination condition
        self.reward, terminated = self._calculate_reward(pipette)
        truncated = False  # Assuming no truncation logic for now

        return self.observation, self.reward, terminated, truncated, {}

    def render(self, mode="human"):
        if self.render_flag:
            self.sim.render(mode)
            
    def _calculate_reward(self, pipette):
        # Calculate the distance to the goal
        distance_to_goal = np.linalg.norm(np.array(self.goal) - pipette)

        # Initialize reward
        reward = 0

        # Progress reward
        progress = self.previous_distance_to_goal - distance_to_goal
        if progress > 0:
            reward += 1.0 * progress  # Scale reward for progress
        else:
            reward -= 0.1  # Penalize stagnation or moving away from goal

        # Directional alignment reward
        direction_to_goal = (np.array(self.goal) - pipette) / (np.linalg.norm(self.goal) + 1e-8)
        if hasattr(self, "previous_action"):
            alignment = np.dot(self.previous_action[:3], direction_to_goal)
            reward += 0.5 * alignment  # Encourage alignment with the goal direction

        # Time penalty to encourage efficiency
        reward -= 0.02

        # Check if the agent has reached the goal
        terminated = distance_to_goal < 0.05
        if terminated:
            reward += 10  # Large reward for reaching the goal

        # Update tracking variables
        self.previous_distance_to_goal = distance_to_goal
        self.previous_action = np.array(self.goal) - pipette  # Approximation of movement vector

        return reward, terminated

    def _drop_ink(self):
        self.ink = 1 - self.ink  # Toggle ink drop state