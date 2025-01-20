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
        self.previous_distance = np.inf  # Track the previous distance to the goal for progress reward

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

        # Randomize goal position
        self.goal_position = np.array([
            np.random.uniform(-0.18700, 0.25300),
            np.random.uniform(-0.17050, 0.21950),
            np.random.uniform(0.16940, 0.28950)
        ], dtype=np.float32)

        # Reset the simulation and get the initial robot position
        observation = self.simulation.reset(num_agents=1)
        robot_position = self.simulation.get_pipette_position(self.simulation.robotIds[0])

        # Combine robot position and goal position
        observation = np.concatenate((robot_position, self.goal_position), axis=0).astype(np.float32)
        self.current_step = 0
        self.previous_distance = np.linalg.norm(robot_position - self.goal_position)  # Track the initial distance
        
        info = {}
        return observation, info

    def compute_reward(self, robot_position):
        distance_to_goal = np.linalg.norm(robot_position - self.goal_position)
        
        # Directional reward to encourage moving towards the goal
        movement_vector = robot_position - self.previous_position
        goal_vector = self.goal_position - self.previous_position
        direction_reward = np.dot(movement_vector, goal_vector) / (np.linalg.norm(goal_vector) + 1e-6)
        
        # Scale rewards
        reward = -distance_to_goal * 100 + direction_reward * 50
        
        # Bonus for reaching the goal
        if distance_to_goal <= 0.001:
            reward += 100
        
        self.previous_distance = distance_to_goal  # Update previous distance for next step
        
        return reward



    def step(self, action):
        self.current_step += 1
        action = np.append(np.array(action, dtype=np.float32), 0)  # Add drop action to match good code format

        # Run simulation step
        observation = self.simulation.run([action])
        robot_position = self.simulation.get_pipette_position(self.simulation.robotIds[0])
        observation = np.concatenate((robot_position, self.goal_position), axis=0).astype(np.float32)
        
        # Compute the reward
        reward = self.compute_reward(robot_position)

        # Check if the goal is reached
        terminated = np.linalg.norm(robot_position - self.goal_position) <= 0.001
        truncated = self.current_step >= self.max_steps
        
        info = {}
        return observation, reward, terminated, truncated, info

    def render(self, mode='human'):
        pass

    def close(self):
        self.simulation.close()
