import numpy as np

from Environment import Environment


def policy_iteration(render=False):
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

    env = Environment(graph, reward)
    v = np.zeros(state_set)
    q_table = np.zeros([state_set, action_set])
    pi = np.full(state_set, 4)
    # pi = np.random.randint(action_set, size=state_set)

    # iteration times
    step = 100

    # policy_iteration
    for k in range(step):
        # print information
        # print(k, ":")
        # print(pi.reshape((n, m)))

        v_backup = v.copy()

        # policy evaluation
        for j in range(1000):
            nv = np.zeros(state_set)
            for state in range(state_set):
                nv[state] = env.acton_reward(state, pi[state]) + gamma * v[env.next_state(state, pi[state])]
            v = nv.copy()
        # policy improvement
        for state in range(state_set):
            for action in range(action_set):
                q_table[state][action] = env.acton_reward(state, action) + gamma * v[env.next_state(state, action)]
            pi[state] = np.argmax(q_table[state])

        # print information
        # print("v:")
        # print(v.reshape((n, m)))
        # print("q_table:")
        # print(q_table)

        state_value_error = sum(abs(v_backup - v))

        # render
        if render:
            env.reset()
            env.explore_by_deterministic_policy(pi, 10, 0.1)

        if state_value_error < 0.01:
            print("converge at %d-th" % k)
            env.reset()
            env.explore_by_deterministic_policy(pi, 100, 0.01)
            break


if __name__ == "__main__":
    policy_iteration()
