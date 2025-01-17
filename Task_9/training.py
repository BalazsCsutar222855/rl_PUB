import wandb
import os
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from new_env import CustomEnv  # Assuming CustomEnv is in this file
from stable_baselines3.common.monitor import Monitor
from clearml import Task
import typing_extensions 
from tensorflow.keras.callbacks import TensorBoard

# Initialize command-line argument parser for hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=0.0001)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--n_steps", type=int, default=1000)  # 1000 steps per iteration
parser.add_argument("--n_epochs", type=int, default=10)
parser.add_argument("--iterations", type=int, default=20)  # Iterations per episode
args = parser.parse_args()

# Set WandB API key and initialize the project
os.environ['WANDB_API_KEY'] = '8afbb298b3eae0f6035d2e3b3bdcadf08ebb1a41'  # Use your actual API key here
wandb.init(project="sb3_custom_env", sync_tensorboard=True)

# Set up the environment
env = CustomEnv()

# Initialize ClearML task for remote training setup
task = Task.init(project_name='Mentor Group M/Group 2', task_name='RL_PPO_CustomEnv')
task.set_base_docker('deanis/2023y2b-rl:latest')  # Set docker image for remote training
task.execute_remotely(queue_name="default")  # Set task to run remotely on ClearML's default queue

# Set up SAC model with the custom environment, command-line arguments, and TensorBoard logging
model = PPO('MlpPolicy', env, verbose=1, 
            learning_rate=args.learning_rate, 
            batch_size=args.batch_size, 
            n_steps=args.n_steps, 
            n_epochs=args.n_epochs, 
            tensorboard_log=f"./ppo_custom_env_tensorboard/{wandb.run.id}/")

# Total steps per episode = iterations * steps per iteration
total_timesteps = args.iterations * 100000

# Train the model for the calculated number of timesteps
model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False, tb_log_name=f"PPO_run_{wandb.run.id}")

# Save the model incrementally after each episode
model.save(f"ppo_model_episode")
    
# Log the model checkpoint to WandB
wandb.save(f"ppo_model_episode.zip")

# Finish WandB logging after training is complete
wandb.finish()
