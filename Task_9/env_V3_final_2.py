import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim_class import Simulation
import random
from collections import deque

class CustomEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, render=True, max_steps=1000):
        super(CustomEnv, self).__init__()

        self.render = render
        self.max_steps = max_steps
        
        self.sim = Simulation(num_agents=1, render=render)
        
        self.action_space = spaces.Box(np.array([-1, -1, -1]), np.array([1, 1, 1]), shape=(3,), dtype=np.float32)

        self.observation_space = spaces.Box(-np.inf, np.inf, (6,), dtype=np.float32)

        self.goal_pos = np.array([0.0, 0.0, 0.0])

        self.current_step = 0


    def step(self, action):
        # Variables
        self.current_step += 1
        goal_pos = self.goal_pos

        # Ensure that action is of the form (x, y, z) and append 0 for drop action
        if len(action) == 3:
            action = np.append(action, 0)  # Append 0 for the drop action
        else:
            raise ValueError("Action must have exactly 3 values for pipette control")

        # Run the simulation with the provided action
        observation = self.sim.run([action])

        # Process the observation and extract the relevant information (pipette position + goal position)
        robot_id = next(iter(observation))  # Assuming single agent
        pipette_pos = np.array(observation[robot_id]['pipette_position'], dtype=np.float32)

        # Combine pipette position and goal position (both are 3D vectors)
        observation = np.concatenate((pipette_pos, self.goal_pos), axis=0)

        # Explicitly cast the observation to float32
        observation = observation.astype(np.float32)

        # Reward function
        reward = -np.linalg.norm(pipette_pos - goal_pos)  # Negative distance


        # Task completion check
        if np.linalg.norm(pipette_pos - goal_pos) < 0.1:
            terminated = True
            reward += 10  # Reward for completing the task
        else:
            terminated = False

        # Truncation check based on max steps
        if self.current_step >= self.max_steps:
            truncated = True
        else:
            truncated = False

        # Combine terminated and truncated into done flag
        done = terminated or truncated

        info = {}  # No additional info in this case

        return observation, reward, done, truncated, info
    
    def render(self, mode='human'):
        pass

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

        self.current_step = 0
        self.goal_pos = np.array([random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)])

        # Obtain the initial observation from the simulation
        observation = self.sim.reset(num_agents=1)

        # Extract relevant information and format it correctly as a numpy array
        # Assuming that the pipette position is part of the observation
        robot_id = next(iter(observation))
        observation = np.array(observation[robot_id]['pipette_position'] + list(self.goal_pos), dtype=np.float32)

        # Return the observation and an info dictionary (empty or containing relevant details)
        info = self.goal_pos 

        return observation, info

    
    def close(self):
        self.sim.close()