import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim_class import Simulation
import random

class CustomEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, render=False, action_noise=0.1, velocity_range=1.0):
        super(CustomEnv, self).__init__()
        
        # Now the render argument is accepted
        self.render = render
        self.action_noise = action_noise  # Hyperparameter for randomness in action
        self.velocity_range = velocity_range  # Hyperparameter for controlling max velocity
        
        self.sim = Simulation(num_agents=1, render=render)

        # Define action space as discrete or continuous depending on the action
        # Let's assume 3 actions: move_x, move_y, move_z, drop_ink (each can have a few discrete options)
        self.action_space = spaces.Discrete(6)  # 6 discrete actions for simplicity, e.g., 3 move directions with 2 possible velocities each

        # Observation space (pipette position + deltas to the goal)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float64)

        self.goal = [0, 0, 0]
        self.reward = 0
        self.observation = None
        self.ink = 0

    def reset(self, seed=None):
        # Reset other environment states
        # Goal position
        self.goal = [random.randint(0, 1), random.randint(0, 1), random.randint(0, 1)]

        # Reset the simulation state
        state = self.sim.reset(num_agents=1)
        robot_id = next(iter(state))  
        pipette = np.array(state[robot_id]['pipette_position'], dtype=np.float32)

        # Delta between pipette and goal
        pipette_delta_x = self.goal[0] - pipette[0]
        pipette_delta_y = self.goal[1] - pipette[1]
        pipette_delta_z = self.goal[2] - pipette[2]

        # Flattening the observation array
        self.observation = np.concatenate([pipette, [pipette_delta_x, pipette_delta_y, pipette_delta_z]])

        info = {}

        return self.observation, info

    def step(self, action):
        # Action effect based on the action number
        # Let's assume action corresponds to movement in X, Y, Z directions with drop command

        # Map discrete action to velocity
        if action == 0:
            velocity_x = random.uniform(-self.velocity_range, self.velocity_range)
            velocity_y = 0
            velocity_z = 0
        elif action == 1:
            velocity_x = 0
            velocity_y = random.uniform(-self.velocity_range, self.velocity_range)
            velocity_z = 0
        elif action == 2:
            velocity_x = 0
            velocity_y = 0
            velocity_z = random.uniform(-self.velocity_range, self.velocity_range)
        elif action == 3:
            velocity_x = random.uniform(-self.velocity_range, self.velocity_range)
            velocity_y = random.uniform(-self.velocity_range, self.velocity_range)
            velocity_z = 0
        elif action == 4:
            velocity_x = random.uniform(-self.velocity_range, self.velocity_range)
            velocity_y = 0
            velocity_z = random.uniform(-self.velocity_range, self.velocity_range)
        else:
            velocity_x = 0
            velocity_y = random.uniform(-self.velocity_range, self.velocity_range)
            velocity_z = random.uniform(-self.velocity_range, self.velocity_range)

        # Add small noise to velocity to avoid deterministic behavior (if noise is enabled)
        if self.action_noise > 0:
            velocity_x += random.uniform(-self.action_noise, self.action_noise)
            velocity_y += random.uniform(-self.action_noise, self.action_noise)
            velocity_z += random.uniform(-self.action_noise, self.action_noise)
        
        # Command for ink dropping (use current ink state)
        drop_command = self.ink

        # Pack the action into the same format as previous
        actions = [[velocity_x, velocity_y, velocity_z, drop_command]]

        # Run the simulation with the given action
        state = self.sim.run(actions)

        robot_id = next(iter(state))  
        pipette = np.array(state[robot_id]['pipette_position'], dtype=np.float32)

        # Update observation
        pipette_delta_x = self.goal[0] - pipette[0]
        pipette_delta_y = self.goal[1] - pipette[1]
        pipette_delta_z = self.goal[2] - pipette[2]

        self.observation = np.concatenate([pipette, [pipette_delta_x, pipette_delta_y, pipette_delta_z]])

        # Calculate reward
        self.reward, terminated = self._calculate_reward(pipette)
        truncated = False  # Task is not truncated in this setup

        # Info (can include diagnostic information if needed)
        self.info = {}

        return self.observation, self.reward, terminated, truncated, self.info

    def _drop_ink(self):
        if self.ink == 1:
            self.ink = 0
        else:
            self.ink = 1

    def _calculate_reward(self, pipette):
        # Calculate distance to goal
        distance_to_goal = np.linalg.norm(np.array(self.goal) - pipette)
        
        # Initialize reward
        reward = 0

        # Progress reward
        if hasattr(self, 'previous_distance_to_goal'):
            progress = self.previous_distance_to_goal - distance_to_goal
            
            # Penalize very small progress
            if progress < 0.01:
                reward -= 0.1  # Small penalty for insufficient progress
            
            # Reward significant progress
            if progress > 0.1:
                reward += 0.5 * progress  # Scale the reward based on progress magnitude

        self.previous_distance_to_goal = distance_to_goal

        # Add a directional reward for aligning movement toward the goal
        direction_to_goal = np.array(self.goal) - pipette
        direction_to_goal = direction_to_goal / np.linalg.norm(direction_to_goal)  # Normalize
        if hasattr(self, 'previous_action'):
            alignment = np.dot(self.previous_action[:3], direction_to_goal)
            reward += 0.1 * alignment  # Small reward for moving in the right direction

        # Add a time penalty to encourage efficiency
        reward -= 0.02  # Time penalty for each step

        # Check if the task is done (reached the goal)
        terminated = distance_to_goal < 0.05

        # Add a large reward for reaching the goal
        if terminated:
            reward += 10  # Large reward for completing the task

        # Save the last action for directional reward calculation
        self.previous_action = np.array([random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)])

        return reward, terminated
