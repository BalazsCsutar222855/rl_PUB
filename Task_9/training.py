import wandb
import os
import argparse
from stable_baselines3 import PPO
from wandb.integration.sb3 import WandbCallback
from clearml import Task
from env_V3_final_2 import CustomEnv  # Assuming CustomEnv is in this file
import tensorboard
from datetime import datetime
import typing_extensions

# Initialize command-line argument parser for hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=0.0001)  # Updated learning rate
parser.add_argument("--batch_size", type=int, default=32)  # Updated batch size
parser.add_argument("--max_steps", type=int, default=1000)  # Updated n_steps
parser.add_argument("--n_steps", type=int, default=2048)  # Updated n_steps
parser.add_argument("--n_epochs", type=int, default=10)  # n_epochs remains the same
parser.add_argument("--iterations_per_episode", type=int, default=20000)  # Iterations per episode
parser.add_argument("--gamma", type=float, default=0.98)  # Updated gamma
parser.add_argument("--ent_coef", type=float, default=0.02)  # Entropy coefficient for exploration
parser.add_argument("--clip_range", type=float, default=0.15)  # Added clip_range
parser.add_argument("--vf_coef", type=float, default=0.5)  # Added value_coefficient
parser.add_argument("--policy", type=str, default="MlpPolicy")  # Policy type (MlpPolicy)
args = parser.parse_args()

os.environ["WANDB_API_KEY"] = "8afbb298b3eae0f6035d2e3b3bdcadf08ebb1a41"  # Replace with your API key

# Set WandB API key and initialize the project
wandb.login()  # This will prompt for a login if needed, using the credentials stored
wandb_session = wandb.init(project="sb3_custom_env", sync_tensorboard=True)

# Set up the environment
env = CustomEnv(render=False, max_steps=args.max_steps)  # Initialize the custom environment

# Initialize ClearML task for remote training setup
task = Task.init(project_name='Mentor Group M/Group 2', task_name='new_reward')
task.set_base_docker('deanis/2023y2b-rl:latest')  # Set docker image for remote training
task.execute_remotely(queue_name="default")  # Set task to run remotely on ClearML's default queue

# Set up PPO model with the custom environment, command-line arguments, and TensorBoard logging
save_path = f"models/{wandb.run.id}"
os.makedirs(save_path, exist_ok=True)

model = PPO(
    args.policy, env, verbose=1,
    learning_rate=args.learning_rate,
    batch_size=args.batch_size,
    n_steps=args.n_steps,
    n_epochs=args.n_epochs,
    gamma=args.gamma,  # Use gamma from command-line args
    ent_coef=args.ent_coef,  # Use entropy coefficient from command-line args
    clip_range=args.clip_range,  # Added clip range for PPO
    vf_coef=args.vf_coef,  # Added value function coefficient for PPO
    tensorboard_log=f"./runs/{wandb.wandb_session.id}/tensorboard/"
)

# Callback for wandb
callback = WandbCallback(
    model_save_freq=100000,
    model_save_path=f"models/{wandb_session.id}",
    verbose=2,)

# Train the model
model.learn(total_timesteps=args.max_steps * args.iterations_per_episode, callback=callback, progress_bar=True, reset_num_timesteps=False,tb_log_name=f"runs/{run.id}")
# Save the model.
model.save(f"models/{wandb_session.id}/{args.max_steps * args.iterations_per_episode}_baseline")
# Save the model to wandb
wandb.save(f"models/{wandb_session.id}/{args.max_steps * args.iterations_per_episode}_baseline")