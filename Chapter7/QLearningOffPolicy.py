import numpy as np

from Environment import Environment, Info

def get_epsilon_greedy_action(pi=4, epsilon=0.0):
    probabilities = np.full(5, epsilon / 5)
    best_action = pi
    probabilities[best_action] += 1 - epsilon
    return np.random.choice(5, p=probabilities)

def q_learning():
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
    target = 17

    # init
    env = Environment(graph, reward)
    v = np.zeros(state_set)
    q_table = np.random.uniform(low=-0.01, high=0.01, size=(state_set, action_set))
    # q_table = np.zeros((state_set, action_set))
    pi = np.full(state_set, 4)

    # iteration times
    step = 1

    for k in range(step):
        print(k, ": ")
        env.reset()
        trajectory = env.explore_by_epsilon_greedy_policy(pi, 100000, 1 / (k + 1), False, 0.001)
        for i in range(len(trajectory)):
            print(i)
            info = trajectory[i]
            state = info.state
            action = info.action
            reward = info.reward
            next_state = info.next_state
            q_table[state][action] = (q_table[state][action] - 0.1 *
                                      (q_table[state][action] - (reward + gamma * max(q_table[next_state]))))
            pi[state] = np.argmax(q_table[state])

    env.reset()
    env.explore_by_deterministic_policy(pi, 20, True, 0.5)


if __name__ == "__main__":
    q_learning()