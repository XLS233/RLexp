import numpy as np
import torch.optim
from torch.utils import data
from tqdm import tqdm, trange

from Environment import Environment, Info
from model import PolicyNet


def reinforce():
    graph = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 1, 2, 1, 0],
        [0, 1, 0, 0, 0]
    ])
    n = np.size(graph, 0)
    m = np.size(graph, 1)
    state_set = n * m
    action_set = 5
    reward = np.array([0, -10, 1, -10])
    gamma = 0.9
    target = 17

    # init
    env = Environment(graph, reward)
    v = np.zeros(state_set)

    policy_net = PolicyNet()
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=0.001)

    epochs = 20000
    learning_rate = 0.001




if __name__ == "__main__":
    reinforce()
