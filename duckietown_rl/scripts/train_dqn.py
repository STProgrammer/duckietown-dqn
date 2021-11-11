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

directory = '../pytorch_models'

env = launch_env()

# Set seedsđ
seed(args.seed)

state_dim = env.observation_space.shape
print("state dim", state_dim)
action_dim = env.action_space.n
print("action dim", action_dim)

# Initialize the DQN model
model = DQN(state_dim, action_dim) # Your Code Here

# Instantiate a replay buffer with max_size = args.replay_buffer_max_size
replay_buffer = ReplayBuffer() # Your Code Here


rewards = []
total_timesteps = 0
episode_num = 0
done = True
episode_reward = None
episode_timesteps = 0
eps = args.eps_start
eps_decay = args.eps_decay
reward = 0
while total_timesteps < args.max_timesteps:

    if done:
        print(f"Done @ {total_timesteps}")
        if total_timesteps != 0:
            print("Replay buffer length is ", len(replay_buffer.storage))
            print(("Total T: %d Episode Num: %d Episode T: %d Reward: %f") % (
                total_timesteps, episode_num, episode_timesteps, episode_reward))

            # Train the model for episode_timesteps iterations
            # Your Code Here
            model.train(replay_buffer, episode_timesteps)
            
            # Append episode reward to the list of rewards
            # Your Code Here
            rewards.append(episode_reward)

            # Save model
            if total_timesteps % args.save_models_freq == 0 and args.save_models:
                model.save(file_name, directory=directory)

        # Reset environment
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_timesteps = 0
        episode_num += 1

        # Decay epsilon
        # Your Code Here
        eps = eps*eps_decay

    # Select action based on epsilon-greedy policy
    # Your Code Here
    if random.random() < eps:
        action = np.random.randint(0, action_dim)
    else:
        action = model.predict(obs) 
        

    # Perform action
    new_obs, reward, done, _ = env.step(action) # Your Code Here

    if episode_timesteps >= args.env_timesteps:
        done = True

    done_bool = 0 if episode_timesteps + 1 == args.env_timesteps else float(done)

    # Update the episode reward
    # Your Code Here
    episode_reward += reward

    # Store experience in replay buffer
    # Your Code Here
    replay_buffer.add(obs, new_obs, action, reward, done)

    # Update obs to the new obs
    # Your Code Here
    obs = new_obs

    episode_timesteps += 1
    total_timesteps += 1


if args.save_models:
    # Save the trained model
    # Your Code Here
    model.save(file_name, directory)



## Plotting results
import matplotlib.pyplot as plt

x = np.linspace(0, episode_num - 1, episode_num - 1)
fig, ax = plt.subplots(figsize=(15,10))

print(x)

print(rewards)

ax.plot(x, rewards, label = 'reward')



# ax.set_xscale('log')

plt.xlabel("Episodes")
plt.ylabel("Rewards")

#plt.ylim(10e-35, 10e+01)
plt.title("Rewards from training DQN")
plt.legend()
plt.grid()
plt.plot()
plt.show()













