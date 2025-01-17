import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim_class import Simulation  # Assuming your custom Simulation class

class OT2Env(gym.Env):
    def __init__(self, render=False, max_steps=10000):
        super(OT2Env, self).__init__()
        self.render = render
        self.max_steps = max_steps
        
        # Create the simulation environment, passing the render flag
        self.sim = Simulation(num_agents=1, render=render)

        # Define action and observation space
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0], dtype=np.float32),  # x, y, z velocities
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),   # x, y, z velocities
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=np.array([-1, -1, -1, -1, -1, -1], dtype=np.float32),  # Observation space can be larger to accommodate the full 3D space
            high=np.array([1, 1, 1, 1, 1, 1], dtype=np.float32),
            dtype=np.float32
        )

        self.steps = 0
        self.goal_position = np.array([0.5, 0.5, 0.2])

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        
        # Reset the simulation environment
        observation = self.sim.reset(num_agents=1)
        robot_id = next(iter(observation))  
        self.state = np.array(observation[robot_id]['pipette_position'], dtype=np.float32)
    
        # Randomize the goal position each time the environment is reset
        self.goal_position = np.random.uniform([-0.2, -0.2, -0.2], [0.2, 0.2, 0.2])  # New random goal position
        
        # Append goal position to the state
        observation = np.concatenate([self.state, self.goal_position])
    
        self.steps = 0
        return observation

    def step(self, action):
        # Proceed with the usual step functionality
        action = np.append(action, 0)  # Adding 0 for drop action
        observation = self.sim.run([action])

        robot_id = next(iter(observation))  
        self.state = np.array(observation[robot_id]['pipette_position'], dtype=np.float32)
        
        observation = np.concatenate([self.state, self.goal_position])

        # Compute reward based on the current position
        reward = self._compute_reward()

        # Goal detection: Terminate when within a threshold distance
        distance_to_goal = np.linalg.norm(self.state[:3] - self.goal_position)
        terminated = distance_to_goal < 0.02 

        truncated = self.steps >= self.max_steps
        info = {}

        self.steps += 1

        return observation, reward, terminated, truncated, info

    def render(self, mode='human'):
        # Check if the render flag is True before attempting to render
        if self.render:
            self.sim.render(mode)  # Assuming the Simulation class has a render method

    def close(self):
        self.sim.close()

    def _compute_reward(self):
        pipette_position = np.array(self.state[:3])
        distance_to_goal = np.linalg.norm(pipette_position - self.goal_position)
        
        # More linear reward
        progress_reward = 1 - (distance_to_goal / np.max(self.observation_space.high[:3] - self.observation_space.low[:3]))
        
        # Less harsh step penalty
        step_penalty = -0.005
        
        # Bonus for getting very close to the goal
        goal_proximity_bonus = 1.0 if distance_to_goal < 0.05 else 0.0
        
        return progress_reward + step_penalty + goal_proximity_bonus