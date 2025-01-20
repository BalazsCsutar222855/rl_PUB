import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim_class import Simulation

class CustomEnv(gym.Env):
    def __init__(self, render=False, max_steps=1000):
        super(CustomEnv, self).__init__()
        self.render_enabled = render
        self.max_steps = max_steps

        # Create the simulation environment
        self.simulation = Simulation(num_agents=1, render=self.render_enabled)

        # Define action and observation spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        
        self.current_step = 0

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

        # Randomize goal position as done in good code
        self.goal_position = np.array([
            np.random.uniform(-0.18700, 0.25300),
            np.random.uniform(-0.17050, 0.21950),
            np.random.uniform(0.16940, 0.28950)
        ], dtype=np.float32)

        # Reset the simulation and get initial robot position
        observation = self.simulation.reset(num_agents=1)
        robot_position = self.simulation.get_pipette_position(self.simulation.robotIds[0])

        # Combine robot position and goal position
        observation = np.concatenate((robot_position, self.goal_position), axis=0).astype(np.float32)
        self.current_step = 0
        
        info = {}
        return observation, info

    def step(self, action):
        self.current_step += 1
        action = np.append(np.array(action, dtype=np.float32), 0)  # Add drop action to match good code format

        # Run simulation step
        observation = self.simulation.run([action])
        robot_position = self.simulation.get_pipette_position(self.simulation.robotIds[0])
        observation = np.concatenate((robot_position, self.goal_position), axis=0).astype(np.float32)
        
        # Calculate reward as in good code
        distance_to_goal = np.linalg.norm(robot_position - self.goal_position)
        reward = -distance_to_goal  # Negative distance for minimization problem
        if distance_to_goal <= 0.001:  # Task completion reward
            reward = 100  # Positive reward for reaching the goal

        terminated = distance_to_goal <= 0.001
        truncated = self.current_step >= self.max_steps
        
        info = {}
        return observation, reward, terminated, truncated, info

    def render(self, mode='human'):
        pass

    def close(self):
        self.simulation.close()
