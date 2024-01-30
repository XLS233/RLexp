import numpy as np

from Environment import Environment, Info


def MC_exploring_starts():
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
    step = 2000

    cnt = np.zeros((state_set, action_set))
    reward = np.zeros((state_set, action_set))

    # MC exploring starts algorithm
    for k in range(step):
        print(k)
        # pi_backup = pi.copy()
        # v_backup = v.copy()

        s_state = np.random.randint(0, state_set)
        s_action = np.random.randint(0, action_set)
        env.reset(env.next_state(s_state, s_action))
        trajectory = [Info(s_state, s_action, env.acton_reward(s_state, s_action), env.next_state(s_state, s_action))]
        trajectory += env.explore_by_deterministic_policy(pi, 25)
        # trajectory += env.explore_by_deterministic_policy(pi, 25)

        flag = np.zeros((state_set, action_set))
        first_reward = np.zeros((state_set, action_set))
        r = 0
        while len(trajectory) > 0:
            r = gamma * r + trajectory[-1].reward
            flag[trajectory[-1].state][trajectory[-1].action] = 1
            first_reward[trajectory[-1].state][trajectory[-1].action] = r
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
        print(pi.reshape((n, m)))

        # policy_change = np.sum(pi != pi_backup)
        # state_value_error = np.sum(abs(v - v_backup))
        # if policy_change == 0 and state_value_error < 0.01:
        #     print("converge at %d-th" % k)
        #     env.reset()
        #     env.explore_by_deterministic_policy(pi, 100, 0.01)
        #     break
        if (k + 1) % 100 == 0:
            env.reset()
            env.explore_by_deterministic_policy(pi, 25, True, 0.01)


if __name__ == "__main__":
    MC_exploring_starts()
