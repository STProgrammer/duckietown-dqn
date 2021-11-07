import numpy as np
import torch
from args import get_dqn_args_test

from dqn import DQN
from env import launch_env

model_name = "DQN"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = get_dqn_args_test()

file_name = "{}_{}".format(
    model_name,
    args.seed
)

env = launch_env()

state_dim = env.observation_space.shape
print(state_dim)
action_dim = env.action_space.n
print(action_dim)

# Initialize model
model = None # Your Code Here

# Load model from file
# Your Code Here

for _ in range(args.episodes):
    done = False
    obs = env.reset()
    steps = 0
    rewards = []
    while True:
        # Render environment
        # Your Code Here

        # Get optimal action according to the DQN model
        # Your Code Here

        # Perform action
        # Your Code Here

        steps += 1

        # Append reward
        # Your Code Here

        if done or steps >= args.env_timesteps:
            break
    print("mean episode reward:", np.mean(rewards))