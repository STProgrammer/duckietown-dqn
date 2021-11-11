import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




# This is code is partly taken and edited from pytorch.org https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
class Network(nn.Module):
    def __init__(self, state_dim, action_dim):

        super(Network, self).__init__()
        # Define the layers of the CNN
        # Your Code Here
        d, w, h = state_dim
        fcls = 16 # Size of first convolution layer
        self.conv1 = nn.Conv2d(d, fcls, kernel_size=5, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(fcls)
        self.conv2 = nn.Conv2d(fcls, fcls*2, kernel_size=5, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(fcls*2)
        self.conv3 = nn.Conv2d(fcls*2, fcls*2, kernel_size=5, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(fcls*2)
        
        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2, padding=1):
            return (size - kernel_size + 2*padding) // stride  + 1
        
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * fcls*2
        
        self.fc_1 = nn.Linear(linear_input_size, 256)
        self.output = nn.Linear(256, action_dim)
        
        

    def forward(self, x):
        # Define the forward pass
        # Your Code Here
        x = x.to(device)
   #     x = torch.tensor(x, dtype=torch.double)
        v = self.conv1(x)
        x = F.relu(self.bn1(v))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.fc_1(x)
        return self.output(x)


class DQN(object):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Define the Q network and the target network (move to the device)
        # Your Code Here
        self.value_net = Network(state_dim, action_dim).double()
        self.value_net.to(device)
        self.target_net = Network(state_dim, action_dim).double()
        self.target_net.to(device)
        self.target_net.eval()
        
        # Use load_state_dict to load the same weight of Q network to the target
        # Your Code Here
        self.target_net.load_state_dict(self.value_net.state_dict())
        

        # Define an optimizer with a learning rate
        # Your Code Here
        self.optimizer = torch.optim.RMSprop(self.value_net.parameters(), lr=0.001, weight_decay=0.5)

        # Define the loss criterion
        # Your Code Here
        self.criterion = nn.MSELoss().to(device)


    def predict(self, state):
        # The input state is a numpy array
        # Just making sure the state has the correct format, otherwise the prediction doesn't work
        assert state.shape[0] == 3

        # Return the action with the highest Q value (action is either 0, 1 or 2 for Left, Right and Straight respectively)
        # Your Code Here
        input_val = torch.from_numpy(state).unsqueeze(0)
        
        output_val = self.value_net(input_val).max(1)[1].item()
        # print("output_val", output_val)
        return output_val
        

    def train(self, replay_buffer, iterations, batch_size=32, discount=0.99):
        for _ in range(iterations):
            # Get a sample from the replay buffer
            # Your Code Here
            sample = replay_buffer.sample(batch_size)

            state = torch.DoubleTensor(sample["state"]).to(device)
            action = torch.LongTensor(sample["action"]).to(device)
            next_state = torch.DoubleTensor(sample["next_state"]).to(device)
            done = torch.DoubleTensor(1 - sample["done"]).to(device)
            reward = torch.DoubleTensor(sample["reward"]).to(device)

            # Compute the target Q value
            # Your Code Here
            
            # print("ACTION INSIDE TRAIN", action)
            # print("STATE INSIDE TRAIN", state)
            # print("NEXT STATE INSIDE TRAIN", next_state)
            # print("REWARD INSIDE TRAIN", reward)
            # print("DONE INSIDE TRAIN", done)

            # TO DO copy state_q
            
            state_q = self.value_net(state)
            with torch.no_grad():
                # state_q = self.value_net(state)
                next_state_max_q = torch.max(self.target_net(next_state), dim=1)[0]
            
            target_q = reward.squeeze() + discount*next_state_max_q * done.squeeze()
            action = action.type(torch.LongTensor).squeeze()
            
            target_state_q = state_q.clone().detach().to(device)
            target_state_q[torch.arange(state_q.shape[0]), action] = target_q
                
            

            # Compute loss
            loss = self.criterion(state_q, target_state_q)
            # Your Code Here

            # Optimizing steps (backward pass, gradient updates ...)
            # Your code here
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            

            # Update target network 
            self.update_target()

    def update_target(self, tau=0.001):
        # Update the frozen target model
        for param, target_param in zip(self.value_net.parameters(), self.target_net.parameters()):            
            new_target_param = tau * target_param + (1 - tau) * param # Your Code Here (Soft update formula)
            target_param.data.copy_(new_target_param)

    def save(self, filename, directory):
        # Saves the Q Network parameters in a file 
        torch.save(self.value_net.state_dict(), '{}/{}_dqn.pth'.format(directory, filename))

    def load(self, filename, directory):
        # Loads the Q Network parameters from a file
        self.value_net.load_state_dict(torch.load('{}/{}_dqn.pth'.format(directory, filename), map_location=device))