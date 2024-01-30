import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange, tqdm

from Environment import Environment, Info


def get_epsilon_greedy_action(pi=4, epsilon=0.0):
    probabilities = np.full(5, epsilon / 5)
    best_action = pi
    probabilities[best_action] += 1 - epsilon
    return np.random.choice(5, p=probabilities)


def sarsa():
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
    reward = np.array([-1, -10, 0, -10])
    gamma = 0.9
    target = 17

    # init
    env = Environment(graph, reward)
    v = np.zeros(state_set)
    # q_table = np.random.uniform(low=-0.01, high=0.01, size=(state_set, action_set))
    q_table = np.zeros((state_set, action_set))
    pi = np.full(state_set, 4)

    # iteration times
    step = 500

    # Sarsa algorithm
    for k in tqdm(range(step), desc="Training", unit="episode"):
        state = 0
        action = get_epsilon_greedy_action(int(pi[state]), 0.1)
        env.reset()
        while state != target:
            reward = env.acton_reward(state, action)
            next_state = env.next_state(state, action)
            next_action = get_epsilon_greedy_action(int(pi[next_state]), 0.1)
            q_table[state][action] = (q_table[state][action] - 0.1 *
                                      (q_table[state][action] - (reward + gamma * q_table[next_state][next_action])))
            pi[state] = np.argmax(q_table[state])
            env.take_action(action, False, 0.01)
            state, action = next_state, next_action

    env.reset()
    env.explore_by_deterministic_policy(pi, 20, True, 0.5)


if __name__ == "__main__":
    sarsa()
