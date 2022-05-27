import numpy as np
import math

from math import sqrt
import random
import torch
from matplotlib import path
from environment import radio_env
from ddpgcode import DDPG
import matplotlib
matplotlib.use("TkAgg")  # Do this before importing pyplot!
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


num_episode = 1
num_iteration = 2000
batch_size = 32



for eps in range(num_episode):
    env = radio_env()

    h = env.SP.H[:, :, 0]
    MRC = h.conj().T @ h * env.SP.power_lin / env.SP.noise

    ddpg = DDPG(env.num_state, env.num_action, env.begin_angle)
    state = env.reset()

    value_result = []
    policy_result = []
    reward_result = []

    for iter in range(num_iteration):
        #print(iter)
        action = ddpg.select_action(state)
        next_state, reward, done = env.step(action)

        ddpg.remember(state, action, reward, next_state, done)
        state = next_state

        if ddpg.memory.__len__() > batch_size:
            minibatch = random.sample(ddpg.memory, batch_size)
            V, P = ddpg.update_parameters(minibatch)
            value_result.append(V)
            policy_result.append(P)

        reward_result.append(reward[0][0])

        if np.abs(reward[0][0] - MRC) < 1e-4:
            break

    h = env.SP.H[:, :, 0]

    plt.axhline(y=(MRC+0.05), color='r')


    plt.plot(reward_result)
    # plt.legend(['MRC', 'DRL'])
    plt.legend(['SVD-based', 'DRL'])

    # plt.plot(value_result)
    # plt.plot(policy_result)
    # plt.legend(['V', 'P'])
    plt.xlabel('Iteration')
    plt.ylabel('Achieved SNR')
    plt.gca().set_ylim(bottom=0)
    plt.show()