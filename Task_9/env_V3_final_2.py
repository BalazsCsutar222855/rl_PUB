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
        """Compute the reward based on the robot's distance to the goal and progress."""
        distance_to_goal = np.linalg.norm(robot_position - self.goal_position)
        
        # Reward for getting closer to the goal
        reward = -distance_to_goal  # Negative distance for minimization problem
        
        # Task completion (goal reached)
        if distance_to_goal <= 0.01:  # Goal threshold
            reward += 100  # Large positive reward for completing the task
        
        self.previous_distance = distance_to_goal  # Update previous distance for next step
        
        return reward

    def step(self, action):
        self.current_step += 1
        action = np.append(np.array(action, dtype=np.float32), 0)

        # Call the environment step function
        observation = self.simulation.run([action]) # Why do we need to pass the action as a list? Think about the simulation class.
        # now we need to process the observation and extract the relevant information, the pipette position, convert it to a numpy array, and append the goal position and make sure the array is of type np.float32
        pipette_position = np.array(observation[f'robotId_{self.simulation.robotIds[0]}']['pipette_position'], dtype=np.float32)



        observation = np.concatenate([pipette_position, self.goal_position], axis=0)
        reward = float(-np.linalg.norm(pipette_position - self.goal_position))
        
        distance = np.linalg.norm(pipette_position - self.goal_position)
        if distance < 0.001:
            terminated = True
            # we can also give the agent a positive reward for completing the task
            reward = float(100)
        else:
            terminated = False
        
        # next we need to check if the episode should be truncated, we can check if the current number of steps is greater than the maximum number of steps
        if self.steps >= self.max_steps:
            truncated = True
        else:
            truncated = False

        info = {} # we don't need to return any additional information

        return observation, reward, terminated, truncated, info

    def render(self, mode='human'):
        pass

    def close(self):
        self.simulation.close()
