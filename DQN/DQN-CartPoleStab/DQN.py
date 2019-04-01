# coding: utf-8

import random
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import copy
from collections import deque
from utils import *

class MLP(nn.Module):
    '''A simple implementation of the multi-layer neural network'''
    def __init__(self, n_input=4, n_output=3, n_h=1, size_h=256):
        '''
        Specify the neural network architecture

        :param n_input: The dimension of the input
        :param n_output: The dimension of the output
        :param n_h: The number of the hidden layer
        :param size_h: The dimension of the hidden layer
        '''
        super(MLP, self).__init__()
        self.n_input = n_input
        self.fc_in = nn.Linear(n_input, size_h)
        self.relu = nn.ReLU()
        assert n_h >= 1, "h must be integer and >= 1"
        self.fc_list = nn.ModuleList()
        for i in range(n_h - 1):
            self.fc_list.append(nn.Linear(size_h, size_h))
        self.fc_out = nn.Linear(size_h, n_output)
        # Initialize weight
        nn.init.uniform_(self.fc_in.weight, -0.1, 0.1)
        nn.init.uniform_(self.fc_out.weight, -0.1, 0.1)
        self.fc_list.apply(self.init_normal)

    def forward(self, x):
        out = x.view(-1, self.n_input)
        out = self.fc_in(out)
        out = self.relu(out)
        for _, layer in enumerate(self.fc_list, start=0):
            out = layer(out)
            out = self.relu(out)
        out = self.fc_out(out)
        return out

    def init_normal(self, m):
        if type(m) == nn.Linear:
            nn.init.uniform_(m.weight, -0.1, 0.1)

class ReplayBuffer(object):
    '''DQN replay buffer'''
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        '''Add samples to the buffer'''
        state      = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        '''Sample from the buffer'''
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self):
        return len(self.buffer)

class Policy(object):
    '''Core implementation of the DQN algorithm'''

    def __init__(self, env,config):
        '''
        Load the configuration setting and specify the environment

        :param env: OpenAI gym style environment
        :param config: (Dictionary) Configuration
        '''
        model_config = config["model_config"]
        self.n_states = 5 #env.observation_space.shape[0]
        self.n_actions = model_config["n_actions"]
        self.use_cuda = model_config["use_cuda"]
        if model_config["load_model"]:
            self.model = torch.load(model_config["model_path"])
        else:
            self.model = MLP(self.n_states, self.n_actions, model_config["n_hidden"],
                             model_config["size_hidden"])
        if self.use_cuda:
            self.Variable = lambda *args, **kwargs: autograd.Variable(*args,**kwargs).cuda()
            self.model=self.model.cuda()
            print("Use CUDA")
        else:
            self.Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs)
            self.model = self.model.cpu()

        training_config = config["training_config"]
        self.current_model = self.model
        self.target_model = copy.deepcopy(self.model)
        self.update_target()

        self.memory_size = training_config["memory_size"]
        self.lr = training_config["learning_rate"]
        self.batch_size = training_config["batch_size"]
        self.gamma = training_config["gamma"]
        self.optimizer = optim.Adam(self.current_model.parameters(),lr= self.lr)
        self.replay_buffer = ReplayBuffer(self.memory_size)

    def act(self, state, epsilon):
        '''
        Choose action based on the epsilon-greedy method

        :param state: (numpy array) The observed state from the environment
        :param epsilon: (float) The probability to choose a random action
        :return: (numpy array) The choosed action
        '''
        if random.random() > epsilon:
            state   = self.Variable(torch.FloatTensor(state).unsqueeze(0))
            q_value = self.model.forward(state)
            action  = q_value.max(1)[1].cpu().detach().numpy(  )  # .data[0]
        else:
            action = np.random.randint(size=(1,) ,low=0, high=self.n_actions)
        return action

    def update_target(self):
        '''Update the target network with current network parameters'''
        self.target_model.load_state_dict(self.current_model.state_dict())


    def update_lr(self, lr):
        '''Update the learning rate'''
        self.lr=lr
        self.optimizer = optim.Adam(self.current_model.parameters(),lr= self.lr)

    def train(self):
        '''
        Sample from the replay buffer and train the current q-network

        :return: The training loss
        '''
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)
        state      = self.Variable(torch.FloatTensor(np.float32(state)))
        next_state = self.Variable(torch.FloatTensor(np.float32(next_state)))
        action     = self.Variable(torch.LongTensor(action))
        reward     = self.Variable(torch.FloatTensor(reward))
        done       = self.Variable(torch.FloatTensor(done))

        q_values      = self.current_model(state)
        next_q_values = self.current_model(next_state)
        next_q_state_values = self.target_model(next_state)
        q_value       = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
        expected_q_value = reward + self.gamma * next_q_value * (1 - done)
        loss = (q_value - self.Variable(expected_q_value.data)).pow(2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def save_model(self, model_path = "storage/test.ckpt"):
        '''Save model to a given path'''
        torch.save(self.model, model_path)
