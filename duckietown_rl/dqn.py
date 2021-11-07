import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Network(nn.Module):
    def __init__(self, state_dim, action_dim):

        super(Network, self).__init__()
        # Define the layers of the CNN
        # Your Code Here

    def forward(self, x):
        # Define the forward pass
        # Your Code Here
        pass

class DQN(object):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Define the Q network and the target network (move to the device)
        # Your Code Here

        # Use load_state_dict to load the same weight of Q network to the target
        # Your Code Here

        # Define an optimizer with a learning rate
        # Your Code Here

        # Define the loss criterion
        # Your Code Here


    def predict(self, state):
        # The input state is a numpy array
        # Just making sure the state has the correct format, otherwise the prediction doesn't work
        assert state.shape[0] == 3

        # Return the action with the highest Q value (action is either 0, 1 or 2 for Left, Right and Straight respectively)
        # Your Code Here
        pass

    def train(self, replay_buffer, iterations, batch_size=32, discount=0.99):
        for _ in range(iterations):
            # Get a sample from the replay buffer
            sample = None # Your Code Here

            state = torch.FloatTensor(sample["state"]).to(device)
            action = torch.LongTensor(sample["action"]).to(device)
            next_state = torch.FloatTensor(sample["next_state"]).to(device)
            done = torch.FloatTensor(1 - sample["done"]).to(device)
            reward = torch.FloatTensor(sample["reward"]).to(device)

            # Compute the target Q value
            # Your Code Here

            # Compute loss
            # Your Code Here

            # Optimizing steps (backward pass, gradient updates ...)
            # Your Code Here

            # Update target network 
            self.update_target()

    def update_target(self, tau=0.001):
        # Update the frozen target model
        for param, target_param in zip(self.value_net.parameters(), self.target_net.parameters()):            
            new_target_param = None # Your Code Here (Soft update formula)
            target_param.data.copy_(new_target_param)

    def save(self, filename, directory):
        # Saves the Q Network parameters in a file 
        torch.save(self.value_net.state_dict(), '{}/{}_dqn.pth'.format(directory, filename))

    def load(self, filename, directory):
        # Loads the Q Network parameters from a file
        self.value_net.load_state_dict(torch.load('{}/{}_dqn.pth'.format(directory, filename), map_location=device))