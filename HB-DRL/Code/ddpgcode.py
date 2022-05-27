import numpy as np
from collections import deque
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

class DDPG(object):
    def __init__(self, num_state, num_action, begin_angle):
        self.tau = 0.01 # soft update of target network
        self.gamma = 0.9 # discount factor
        self.memory = deque(maxlen=1000)


        self.actor = Actor(num_state, num_action, begin_angle)
        self.actor_target = Actor(num_state, num_action, begin_angle)
        self.actor_optim = Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(num_state, num_action)
        self.critic_target = Critic(num_state, num_action)
        self.critic_optim = Adam(self.critic.parameters(), lr=1e-3)


        hard_update(self.actor_target, self.actor)  # Make sure target is with the same weights
        hard_update(self.critic_target, self.critic)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        # Make sure we restrict memory size to specified limit
        if len(self.memory) > 2000:
            self.memory.pop(0)

    def select_action(self, state):
        self.actor.eval()
        input = torch.flatten(torch.Tensor(state.astype(np.float32)))
        mu = self.actor(input)

        return mu

    def update_parameters(self, minibatch):
        batch = minibatch.__len__()
        states = torch.Tensor([minibatch[i][0] for i in range(batch)]).reshape(batch, -1)
        actions = torch.stack([minibatch[i][1] for i in range(batch)]) ## already a Tensor
        rewards = torch.Tensor([minibatch[i][2] for i in range(batch)]).reshape(batch, -1)
        next_states = torch.Tensor([minibatch[i][3] for i in range(batch)]).reshape(batch, -1)

        ## Gets target
        next_actions = self.actor_target(next_states)
        next_state_action_values = self.critic_target(next_states, next_actions)
        value_target = rewards + (self.gamma  * next_state_action_values)


        self.critic_optim.zero_grad()

        value = self.critic(states, actions) ## batch!!!!

        value_loss = F.mse_loss(value, value_target)
        value_loss.backward(retain_graph=True)
        self.critic_optim.step()

        self.actor_optim.zero_grad()

        policy_loss = -self.critic(states, self.actor(states)) # gradient ascend

        policy_loss = policy_loss.mean()
        policy_loss.backward(retain_graph=True)
        self.actor_optim.step()

        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        return value_loss.item(), policy_loss.item()



class Actor(nn.Module):
    def __init__(self,num_state, num_action, begin_angle):
        super(Actor, self).__init__()
        self.begin_angle = begin_angle
        self.fc1 = nn.Linear(num_state,200)
        self.fc2 = nn.Linear(200,200)
        self.fc3 = nn.Linear(200,200)
        self.fc4 = nn.Linear(200,num_action)
        self.dropout = torch.nn.Dropout(p=0.8, inplace=False)


    def forward(self, x):
        in_size = x.size(0)  # batch
        x = self.fc1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        #x = x.view(in_size, -1)
        x = self.fc4(x)
        # todo: normalization
        #x[:][self.begin_angle:] = F.tanh(x[:][self.begin_angle:])

        return x



class Critic(nn.Module):
    def __init__(self,num_state, num_action):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(num_state,200)
        self.fc2 = nn.Linear(200 + num_action,200)
        self.fc3 = nn.Linear(200,1)
        self.dropout = torch.nn.Dropout(p=0.8, inplace=False)


    def forward(self, inputs, actions):
        x = inputs
        x = self.fc1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = torch.cat((x, actions),1)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x



