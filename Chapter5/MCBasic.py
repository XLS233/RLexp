import numpy as np

from Environment import Environment, Info


def MC_basic(render=False):
    # basic information
    # graph = np.array([
    #     [0, 0, 0],
    #     [0, 0, 1],
    #     [1, 0, 2]
    # ])
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
    # pi = np.array([0, 2, 1, 1, 2, 2, 1, 1, 4])
    pi = np.full(state_set, 4)
    # pi = np.random.randint(0, 5, size=state_set)

    # iteration times
    step = 100

    # MC_basic_algorithm, converge at 75-th
    for k in range(step):
        print(k)
        npi = np.zeros(state_set)

        # print(v.reshape((n, m)))

        for state in range(state_set):
            for action in range(action_set):
                # get trajectory by pi
                env.reset(env.next_state(state, action))
                trajectory = [Info(state, action, env.acton_reward(state, action), env.next_state(state, action))]
                trajectory += env.explore_by_deterministic_policy(pi, k)
                r = 0
                while len(trajectory) > 0:
                    r = gamma * r + trajectory[-1].reward
                    trajectory.pop()
                # r = gamma * r + env.acton_reward(state, action)
                # q(state, action)
                q_table[state][action] = r
            npi[state] = np.argmax(q_table[state])
            v[state] = np.max(q_table[state])

        policy_change = np.sum(pi != npi)
        print(policy_change)
        pi = npi.copy()
        print(pi.reshape((n, m)))

        if policy_change == 0:
            print("converge at %d-th" % k)
            env.reset()
            env.explore_by_deterministic_policy(pi, 100, 0.01)
            break

    env.reset()
    env.explore_by_deterministic_policy(pi, 100, True, 1)


if __name__ == "__main__":
    MC_basic()
