import numpy as np

from Environment import Environment, Info


def calc_state_value(epsilon=0.0):
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
    q_table = np.full((state_set, action_set), 0)
    pi = np.array([1, 1, 1, 1, 2, 0, 0, 1, 1, 2, 0, 3, 2, 1, 2, 0, 1, 4, 3, 2, 0, 1, 0, 3, 3])

    m_pi = np.zeros((state_set, action_set))
    m_reward = np.zeros((state_set, action_set))
    m_next_state_value = np.zeros((state_set, action_set))

    for state in range(state_set):
        for action in range(action_set):
            if action == pi[state]:
                m_pi[state][action] = 1 - (5 - 1) / 5 * epsilon
            else:
                m_pi[state][action] = 1 / 5 * epsilon
            m_reward[state][action] = env.acton_reward(state, action)

    step = 1000
    for j in range(step):
        for state in range(state_set):
            for action in range(action_set):
                m_next_state_value[state][action] = v[env.next_state(state, action)].copy()
        nv = np.zeros(state_set)
        for state in range(state_set):
            nv[state] = sum(m_pi[state] * (m_reward[state] + gamma * m_next_state_value[state]))
        v = nv.copy()

    print(v.reshape(n, m))


def MC_epsilon_greedy():
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
    q_table = np.full((state_set, action_set), -np.inf)
    pi = np.full(state_set, 4)

    # iteration times
    step = 1500

    cnt = np.zeros((state_set, action_set))
    reward = np.zeros((state_set, action_set))

    # MC epsilon greedy algorithm
    for k in range(step):
        print(k)

        s_state = np.random.randint(0, state_set)
        s_action = np.random.randint(0, action_set)
        env.reset(env.next_state(s_state, s_action))
        trajectory = [Info(s_state, s_action, env.acton_reward(s_state, s_action), env.next_state(s_state, s_action))]
        trajectory += env.explore_by_epsilon_greedy_policy(pi, 25, 0)

        flag = np.zeros((state_set, action_set))
        first_reward = np.zeros((state_set, action_set))
        r = 0
        while len(trajectory) > 0:
            r = gamma * r + trajectory[-1].reward
            flag[trajectory[-1].state][trajectory[-1].action] += 1
            first_reward[trajectory[-1].state][trajectory[-1].action] += r
            trajectory.pop()

        cnt += flag
        reward += first_reward
        for state in range(state_set):
            for action in range(action_set):
                if cnt[state][action] == 0:
                    q_table[state][action] = -np.inf
                else:
                    q_table[state][action] = reward[state][action] / cnt[state][action]
            pi[state] = np.argmax(q_table[state])
            v[state] = np.max(q_table[state])

        print(v.reshape((n, m)))
        # print(pi.reshape((n, m)))

        # if (k + 1) % 100 == 0:
        #     env.reset()
        #     env.explore_by_deterministic_policy(pi, 25, True, 0.01)
    env.reset()
    env.explore_by_deterministic_policy(pi, 25, True, 1)


if __name__ == "__main__":
    MC_epsilon_greedy()
    # calc_state_value(0.5)
