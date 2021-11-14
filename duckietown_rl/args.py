import argparse
import sys

def get_dqn_args_train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=2502, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--max_timesteps", default=20000, type=float)  # Max time steps to run environment for
    parser.add_argument("--save_models", action="store_true", default=True)  # Whether or not the model is saved
    parser.add_argument("--save_models_freq", default=10000, type=int)  # When to save the model
    parser.add_argument("--batch_size", default=16, type=int)  # Batch size
    parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)  # Target network update rate
    parser.add_argument("--eps_start", default=1.0, type=float)  # Epsilon greedy start value
    parser.add_argument("--eps_end", default=0.01, type=float)  # Epsilon greedy end value
    parser.add_argument("--eps_decay", default=0.995, type=float)  # Epsilon greedy decay
    parser.add_argument("--env_timesteps", default=500, type=int)  # Maximum timesteps of one episode
    parser.add_argument("--replay_buffer_max_size", default=5000, type=int)  # Maximum number of steps to keep in the replay buffer

    return parser.parse_args()


def get_dqn_args_test():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=2502, type=int)  # Inform the test what seed was used in training
    parser.add_argument("--episodes", default=10, type=int) # Number of Episodes
    parser.add_argument("--env_timesteps", default=3000, type=int)  # Maximum timesteps of one episode
    
    return parser.parse_args()
