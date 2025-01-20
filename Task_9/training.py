import wandb
import os
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from clearml import Task
from env_V3_final_2 import CustomEnv  # Assuming CustomEnv is in this file
import tensorboard
from datetime import datetime
import typing_extensions

# Initialize command-line argument parser for hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=0.0003)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--n_steps", type=int, default=1000)  # 1000 steps per iteration
parser.add_argument("--n_epochs", type=int, default=10)
parser.add_argument("--episodes", type=int, default=10)  # Total number of episodes
parser.add_argument("--iterations_per_episode", type=int, default=20000)  # Iterations per episode
parser.add_argument("--gamma", type=float, default=0.98)  # Discount factor
parser.add_argument("--ent_coef", type=float, default=0.02)  # Entropy coefficient for exploration
args = parser.parse_args()

os.environ["WANDB_API_KEY"] = "8afbb298b3eae0f6035d2e3b3bdcadf08ebb1a41" 

# Set WandB API key and initialize the project
wandb.login()  # This will prompt for a login if needed, using the credentials stored
wandb.init(project="sb3_custom_env", sync_tensorboard=True)

# Set up the environment
env = CustomEnv(render=False, max_steps=args.n_steps)  # Initialize the custom environment

# Initialize ClearML task for remote training setup
task = Task.init(project_name='Mentor Group M/Group 2', task_name='help')
task.set_base_docker('deanis/2023y2b-rl:latest')  # Set docker image for remote training
task.execute_remotely(queue_name="default")  # Set task to run remotely on ClearML's default queue

# Set up PPO model with the custom environment, command-line arguments, and TensorBoard logging
save_path = f"models/{wandb.run.id}"
os.makedirs(save_path, exist_ok=True)

model = PPO(
    'MlpPolicy', env, verbose=1,
    learning_rate=args.learning_rate,
    batch_size=args.batch_size,
    n_steps=args.n_steps,
    n_epochs=args.n_epochs,
    gamma=args.gamma,  # Use gamma from command-line args
    ent_coef=args.ent_coef,  # Use entropy coefficient from command-line args
    tensorboard_log=f"./runs/{wandb.run.id}/tensorboard/"
)

# Training loop for a specific number of episodes
for episode in range(1, args.episodes + 1):
    print(f"Starting episode {episode}/{args.episodes}")

    # Total steps per episode = iterations * steps per iteration
    total_timesteps = args.iterations_per_episode * args.n_steps  # 200,000 iterations * 1,000 steps/iteration

    # Train the model for the calculated number of timesteps
    model.learn(
        total_timesteps=total_timesteps,
        reset_num_timesteps=False,
        tb_log_name=f"PPO_run_{wandb.run.id}_episode_{episode}",
        callback=BaseCallback(
            model_save_freq=10000,
            model_save_path=save_path,
            verbose=2
        )
    )

    # Generate a timestamp for unique model file naming
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')

    # Save the model incrementally after each episode
    model.save(f"{save_path}/ppo_model_episode_{episode}_{timestamp}")
    
    # Log the model checkpoint to WandB
    wandb.save(f"{save_path}/ppo_model_episode_{episode}_{timestamp}.zip")

# Finish WandB logging after training is complete
wandb.finish()
