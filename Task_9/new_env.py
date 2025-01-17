import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim_class import Simulation
import random
from collections import deque

class CustomEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, render=False, max_steps=1000):
        super(CustomEnv, self).__init__()
        
        self.render = render
        self.sim = Simulation(num_agents=1, render=render)

        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float64)

        self.goal = [0, 0, 0]
        self.reward = 0
        self.observation = None
        self.ink = 0

        # Step counter and maximum steps
        self.current_step = 0
        self.max_steps = max_steps

    def reset(self, seed=None):
        # Reset the step counter
        self.current_step = 0

        # Reset other environment states
        self.goal = [random.randint(0, 1), random.randint(0, 1), random.randint(0, 1)]

        state = self.sim.reset(num_agents=1)
        robot_id = next(iter(state))  
        pipette = np.array(state[robot_id]['pipette_position'], dtype=np.float32)

        pipette_delta_x = self.goal[0] - pipette[0]
        pipette_delta_y = self.goal[1] - pipette[1]
        pipette_delta_z = self.goal[2] - pipette[2]

        self.observation = np.concatenate([pipette, [pipette_delta_x, pipette_delta_y, pipette_delta_z]])

        info = {}
        return self.observation, info

    def step(self, action):
        # Increment step counter
        self.current_step += 1

        # Define directions for action (stay within range -1 to 1 for velocity)
        # The action space might be: 0 = move left, 1 = move right, etc.
        if action == 0:
            velocity_x, velocity_y, velocity_z = -0.1, 0, 0  # Move left
        elif action == 1:
            velocity_x, velocity_y, velocity_z = 0.1, 0, 0  # Move right
        elif action == 2:
            velocity_x, velocity_y, velocity_z = 0, -0.1, 0  # Move down
        elif action == 3:
            velocity_x, velocity_y, velocity_z = 0, 0.1, 0  # Move up
        elif action == 4:
            velocity_x, velocity_y, velocity_z = 0, 0, -0.1  # Move backward
        elif action == 5:
            velocity_x, velocity_y, velocity_z = 0, 0, 0.1  # Move forward

        # Allow for small random variance to encourage exploration
        velocity_x += random.uniform(-0.05, 0.05)
        velocity_y += random.uniform(-0.05, 0.05)
        velocity_z += random.uniform(-0.05, 0.05)

        # Define actions based on velocity and drop command
        actions = [[velocity_x, velocity_y, velocity_z, self.ink], [velocity_x, velocity_y, velocity_z, self.ink]]
        
        # Execute action in the simulation
        state = self.sim.run(actions)

        # Extract pipette position from the simulation state
        robot_id = next(iter(state))
        pipette = np.array(state[robot_id]['pipette_position'], dtype=np.float32)

        # Update observation with new position and deltas to goal
        pipette_delta_x = self.goal[0] - pipette[0]
        pipette_delta_y = self.goal[1] - pipette[1]
        pipette_delta_z = self.goal[2] - pipette[2]

        self.observation = np.concatenate([pipette, [pipette_delta_x, pipette_delta_y, pipette_delta_z]])

        # Calculate reward and check termination condition
        self.reward, terminated = self._calculate_reward(pipette)

        # Check if maximum steps have been reached
        if self.current_step >= self.max_steps:
            terminated = True

        truncated = False

        # Log for debugging
        print(f"Step {self.current_step}:")
        print(f"  Pipette Position: {pipette}")
        print(f"  Goal Position: {self.goal}")
        print(f"  Distance to Goal: {np.linalg.norm(np.array(self.goal) - pipette)}")
        print(f"  Reward: {self.reward}")

        return self.observation, self.reward, terminated, truncated, {}



    def _calculate_reward(self, pipette):
        distance_to_goal = np.linalg.norm(np.array(self.goal) - pipette)
        
        # Base reward: penalize distance to the goal
        reward = -distance_to_goal

        # Reward for progress (if applicable)
        if hasattr(self, 'previous_distance_to_goal'):
            reward += self.previous_distance_to_goal - distance_to_goal
        self.previous_distance_to_goal = distance_to_goal

        # Bonus for proximity to the goal
        if distance_to_goal < 1.5:
            reward += 5 * (1.5 - distance_to_goal)

        # Termination condition: reached the goal
        terminated = distance_to_goal < 1
        if terminated:
            reward += 10  # Big reward for success

        return reward, terminated
