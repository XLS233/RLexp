import numpy as np
from Environment import Environment


def random_graph(size=10):
    graph = np.random.randint(2, size=(size, size))
    x = np.random.randint(0, size)
    y = np.random.randint(0, size)
    graph[x][y] = 2
    return graph


def value_iteration(render=False):
    # basic information
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
    reward = np.array([0, -10, 1, -1])
    gamma = 0.9

    # init
    env = Environment(graph, reward)
    v = np.zeros(state_set)
    q_table = np.zeros([state_set, action_set])
    pi = np.zeros(state_set)

    # iteration times
    step = 1000

    # value iteration algorithm
    for k in range(step):
        # new_v: v_(K+1)
        nv = np.zeros(state_set)
        for state in range(state_set):
            for action in range(action_set):
                # update q_table
                q_table[state][action] = (env.acton_reward(state, action) + gamma * v[env.next_state(state, action)])
            # update policy
            pi[state] = int(np.argmax(q_table[state]))
            # update state value function
            nv[state] = np.max(q_table[state])

        state_value_error = sum(abs(nv - v))
        v = nv.copy()

        # render
        if render:
            env.reset()
            env.explore_by_deterministic_policy(pi, 100, 0.01)

        if state_value_error < 0.01:
            print("converge at %d-th" % k)
            env.reset()
            env.explore_by_deterministic_policy(pi, 100, 0.01)
            break


if __name__ == "__main__":
    value_iteration()

"""
graph = np.array([
    [0, 1],
    [0, 2]
])
graph = np.array([
    [0, 0, 0],
    [0, 0, 1],
    [1, 0, 2]
])
graph = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 1, 2, 1, 0],
    [0, 1, 0, 0, 0]
])
"""

