import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim_class import Simulation

class CustomEnv(gym.Env):
    def __init__(self, render=False, max_steps=5000000):
        super(CustomEnv, self).__init__()
        self.render_enabled = render
        self.max_steps = max_steps

        # Create the simulation environment
        self.simulation = Simulation(num_agents=1, render=self.render_enabled)

        # Define goal position ranges
        self.goal_x_min, self.goal_x_max = -0.187, 0.2531
        self.goal_y_min, self.goal_y_max = -0.1705, 0.2195
        self.goal_z_min, self.goal_z_max = 0.1195, 0.2895

        # Define action and observation spaces
        # Change action space to 4-dimensional as in the good code
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        self.observation_space = spaces.Box(
            low=np.array([self.goal_x_min, self.goal_y_min, self.goal_z_min, -self.goal_x_max, -self.goal_y_max, -self.goal_z_max], dtype=np.float32),
            high=np.array([self.goal_x_max, self.goal_y_max, self.goal_z_max, self.goal_x_max, self.goal_y_max, self.goal_z_max], dtype=np.float32),
            dtype=np.float32
        )
        
        self.current_step = 0

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

        # Randomize goal position as done in good code
        goal_x = np.random.uniform(self.goal_x_min, self.goal_x_max)
        goal_y = np.random.uniform(self.goal_y_min, self.goal_y_max)
        goal_z = np.random.uniform(self.goal_z_min, self.goal_z_max)
        self.goal_position = np.array([goal_x, goal_y, goal_z], dtype=np.float32)

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
            reward += 100  # Positive reward for reaching the goal

        terminated = distance_to_goal <= 0.001
        if terminated:
            reward += 10  # Additional reward for completion

        truncated = self.current_step >= self.max_steps
        
        info = {}
        return observation, reward, terminated, truncated, info

    def render(self, mode='human'):
        pass

    def close(self):
        self.simulation.close()
