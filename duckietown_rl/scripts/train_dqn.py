import pickle
import random
import resource
import gym_duckietown
import numpy as np
import torch
import gym
import os

from args import get_dqn_args_train
from dqn import DQN
from utils import seed, ReplayBuffer
from env import launch_env


model_name = "DQN"

print(f"Using {'cuda' if torch.cuda.is_available() else 'cpu'}")

args = get_dqn_args_train()

file_name = "{}_{}".format(
    model_name,
    str(args.seed),
)

env = launch_env()

# Set seeds
seed(args.seed)

state_dim = env.observation_space.shape
print(state_dim)
action_dim = env.action_space.n
print(action_dim)

# Initialize the DQN model
model = None # Your Code Here

# Instantiate a replay buffer with max_size = args.replay_buffer_max_size
replay_buffer = None # Your Code Here


rewards = []
total_timesteps = 0
episode_num = 0
done = True
episode_reward = None
eps = args.eps_start
while total_timesteps < args.max_timesteps:

    if done:
        print(f"Done @ {total_timesteps}")
        if total_timesteps != 0:
            print("Replay buffer length is ", len(replay_buffer.storage))
            print(("Total T: %d Episode Num: %d Episode T: %d Reward: %f") % (
                total_timesteps, episode_num, episode_timesteps, episode_reward))

            # Train the model for episode_timesteps iterations
            # Your Code Here

            # Append episode reward to the list of rewards
            # Your Code Here

            # Save model
            if total_timesteps % args.save_models_freq == 0 and args.save_models:
                model.save(file_name, directory="./pytorch_models")

        # Reset environment
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_timesteps = 0
        episode_num += 1

        # Decay epsilon
        # Your Code Here

    # Select action based on epsilon-greedy policy
    # Your Code Here
    if random.random() < eps:
        pass
    else:
        pass

    # Perform action
    new_obs, reward, done, _ = (None, None, None, None) # Your Code Here

    if episode_timesteps >= args.env_timesteps:
        done = True

    done_bool = 0 if episode_timesteps + 1 == args.env_timesteps else float(done)

    # Update the episode reward
    # Your Code Here

    # Store experience in replay buffer
    # Your Code Here

    # Update obs to the new obs
    # Your Code Here

    episode_timesteps += 1
    total_timesteps += 1


if args.save_models:
    # Save the trained model
    # Your Code Here
    pass
