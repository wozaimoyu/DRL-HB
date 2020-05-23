import numpy as np
import math

from math import sqrt
import random
import torch
from matplotlib import path
from environment import radio_env
from ddpgcode import DDPG
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

num_episode = 1
num_iteration = 5000
batch_size = 8

for eps in range(num_episode):
    env = radio_env()
    ddpg = DDPG(env.num_state, env.num_action)
    state = env.reset()

    value_result = []
    policy_result = []
    reward_result = []

    for iter in range(num_iteration):
        action = ddpg.select_action(state)
        next_state, reward, done = env.step(action)

        ddpg.remember(state, action, reward, next_state, done)
        state = next_state

        if ddpg.memory.__len__() > batch_size:
            minibatch = random.sample(ddpg.memory, batch_size)
            V, P = ddpg.update_parameters(minibatch)
            value_result.append(V)
            policy_result.append(P)

        reward_result.append(reward)

    plt.plot(reward_result)
    # plt.plot(value_result)
    # plt.plot(policy_result)
    # plt.legend(['V', 'P'])
    plt.show()