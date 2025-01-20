import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim_class import Simulation

class CustomEnv(gym.Env):
    def __init__(self, render=False, max_steps=5000000):
        super(CustomEnv, self).__init__()
        self.render = render
        self.max_steps = max_steps

        # Create the simulation environment
        self.sim = Simulation(num_agents=1, render=self.render)

        # Define goal position ranges
        self.x_min, self.x_max = -0.187, 0.2531
        self.y_min, self.y_max = -0.1705, 0.2195
        self.z_min, self.z_max = 0.1195, 0.2895

        # Define action and observation spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([self.x_min, self.y_min, self.z_min, -self.x_max, -self.y_max, -self.z_max], dtype=np.float32),
            high=np.array([self.x_max, self.y_max, self.z_max, self.x_max, self.y_max, self.z_max], dtype=np.float32),
            dtype=np.float32
        )
        
        self.steps = 0

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

        # Randomize goal position
        x = np.random.uniform(self.x_min, self.x_max)
        y = np.random.uniform(self.y_min, self.y_max)
        z = np.random.uniform(self.z_min, self.z_max)
        self.goal_pos = np.array([x, y, z], dtype=np.float32)

        # Reset the simulation and get initial pipette position
        observation = self.sim.reset(num_agents=1)
        pipette_pos = self.sim.get_pipette_position(self.sim.robotIds[0])

        # Combine pipette position and goal position
        observation = np.concatenate((pipette_pos, self.goal_pos), axis=0).astype(np.float32)
        self.steps = 0
        
        info = {}
        return observation, info

    def step(self, action):
        self.steps += 1
        action = np.append(np.array(action, dtype=np.float32), 0)

        # Run simulation step
        observation = self.sim.run([action])
        pipette_pos = self.sim.get_pipette_position(self.sim.robotIds[0])
        observation = np.concatenate((pipette_pos, self.goal_pos), axis=0).astype(np.float32)
        
        # Calculate reward
        distance = np.linalg.norm(pipette_pos - self.goal_pos)
        reward = -distance
        reward -= 0.01 * self.steps

        terminated = distance <= 0.001
        if terminated:
            reward += 10

        truncated = self.steps >= self.max_steps
        
        info = {}
        return observation, reward, terminated, truncated, info

    def render(self, mode='human'):
        pass

    def close(self):
        self.sim.close()
