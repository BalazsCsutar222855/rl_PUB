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
        self.previous_position = None  # Track the robot's position for directional rewards


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
        self.previous_position = robot_position  # Initialize the robot's previous position

        info = {}
        return observation, info


    def compute_reward(self, robot_position):
        """Compute the reward based on the robot's distance to the goal and progress."""
        distance_to_goal = np.linalg.norm(robot_position - self.goal_position)

        # Progressive reward for getting closer to the goal
        progress_reward = (self.previous_distance - distance_to_goal) * 100  # Encourages steady progress

        # Directional reward (dot product to encourage movement toward goal)
        movement_vector = robot_position - self.previous_position
        goal_vector = self.goal_position - self.previous_position
        direction_reward = np.dot(movement_vector, goal_vector) / (np.linalg.norm(goal_vector) + 1e-6)
        direction_reward *= 50  # Scale for impact

        # Efficiency reward (penalize unnecessary movement)
        efficiency_penalty = -np.linalg.norm(movement_vector) * 0.1  # Penalize excessive movement

        # Task completion reward
        completion_reward = 100 if distance_to_goal <= 0.001 else 0

        # Small time penalty to encourage faster task completion
        time_penalty = -0.01  

        # Total reward
        reward = progress_reward + direction_reward + efficiency_penalty + completion_reward + time_penalty

        # Update previous values for next step
        self.previous_distance = distance_to_goal
        self.previous_position = robot_position

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
