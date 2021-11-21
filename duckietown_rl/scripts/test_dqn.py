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

directory = '../pytorch_models'

env = launch_env()

state_dim = env.observation_space.shape
print(state_dim)
action_dim = env.action_space.n
print(action_dim)

# Initialize model
model = DQN(state_dim, action_dim) # Your Code Here

# Load model from file
model.load(file_name, directory)

mean_step_rewards_for_all = list()
sum_rewards_for_all = list()

for _ in range(args.episodes):
    done = False
    obs = env.reset()
    steps = 0
    rewards = []
    sum_reward = 0
    while True:
        # Render environment
        env.render()

        # Get optimal action according to the DQN model
        action = model.predict(obs)

        # Perform action
        new_obs, reward, done, _ = env.step(action)
        obs = new_obs

        steps += 1

        # Append reward
        rewards.append(reward)
        sum_reward += reward
        
        

        if done or steps >= args.env_timesteps:
            break
    sum_rewards_for_all.append(sum_reward)
    
    mean_step_reward = np.mean(rewards)
    mean_step_rewards_for_all.append(mean_step_reward)
    print("Mean episode reward:", mean_step_reward)
   
    
print("Mean episode rewards for each episode", mean_step_rewards_for_all)
print("Mean episode reward of all episodes", np.mean(mean_step_rewards_for_all))

print("Sum episode reward for each episode", sum_rewards_for_all)
print("Mean sum episode reward", np.mean(sum_rewards_for_all))
