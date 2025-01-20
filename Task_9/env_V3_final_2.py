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

    def compute_reward(self, action):
        """
        Computes the reward based on the action taken and the distance from the target.
        """
        # Augment the action with a zero for simulation compatibility
        action_with_drop = np.append(np.array(action, dtype=np.float32), 0)

        # Execute the simulation and retrieve the observation
        result = self.simulation.run([action_with_drop])
        current_position = self.simulation.get_pipette_position(self.simulation.robotIds[0])

        # Determine how far the pipette is from the goal
        dist_to_goal = np.linalg.norm(current_position - self.goal_position)

        # If this is the first step, store the initial distance and previous distance
        if self.previous_distance is np.inf:
            self.previous_distance = dist_to_goal
            self.initial_dist = dist_to_goal

        # Calculate how much closer we got to the goal
        step_progress = self.previous_distance - dist_to_goal
        reward_for_progress = step_progress * 15  # Reward scaling

        # Update the previous distance for the next step
        self.previous_distance = dist_to_goal

        # Reward bonuses based on reaching specific milestones
        reward_milestone = 0
        threshold_milestones = [0.2, 0.4, 0.6, 0.7, 0.75, 0.85, 0.9, 0.95, 1.0]
        for milestone in threshold_milestones[:]:
            if (self.previous_distance > milestone * self.initial_dist) and (dist_to_goal <= milestone * self.initial_dist):
                reward_milestone += 22 * milestone  # Milestone reward scaling
                threshold_milestones.remove(milestone)

        # Final reward calculation
        total_reward = reward_for_progress + reward_milestone - 0.03  # Small penalty for every step

        return total_reward, dist_to_goal

    def step(self, action):
        """
        Executes one step in the environment, calculating the reward and determining if the task is complete.
        """
        # Calculate the reward for the action
        reward, current_distance = self.compute_reward(action)

        # Check if the goal is reached (distance to the goal is small enough)
        goal_threshold = 0.001
        if current_distance <= goal_threshold:
            reward += 120  # Large bonus for reaching the goal
            terminated = True
        else:
            terminated = False

        # Track if the simulation should be truncated
        if self.current_step >= self.max_steps:
            truncated = True
        else:
            truncated = False

        # Combine termination and truncation into done flag
        done = terminated or truncated

        # Get the current observation
        robot_position = self.simulation.get_pipette_position(self.simulation.robotIds[0])
        observation = np.concatenate((robot_position, self.goal_position), axis=0).astype(np.float32)
        
        # Info can be an empty dictionary or any additional info
        info = {}

        # Return the observation, reward, done, truncated, and any additional info
        return observation, reward, done, truncated, info

    def render(self, mode='human'):
        pass

    def close(self):
        self.simulation.close()
