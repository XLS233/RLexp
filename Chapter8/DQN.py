import numpy as np
import torch.optim
from torch.utils import data
from tqdm import tqdm, trange

from Environment import Environment, Info
from model import QNET


def get_data_loader(episode, batch_size=64, env=None):
    reward = []
    state_action = []
    next_state = []
    for info in episode:
        reward.append(info.reward)
        action = info.action
        x, y = env.state_to_pos(info.state)
        state_action.append((x, y, action))
        x, y = env.state_to_pos(info.next_state)
        next_state.append((x, y))
    reward = torch.tensor(reward).reshape((-1, 1))
    state_action = torch.tensor(state_action)
    next_state = torch.tensor(next_state)
    data_array = (state_action, reward, next_state)
    dataset = data.TensorDataset(*data_array)
    return data.DataLoader(dataset, batch_size, shuffle=True, drop_last=False)


def dqn():
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

    # init
    env = Environment(graph, reward)
    v = np.zeros(state_set)
    # q_table = np.random.uniform(low=-0.01, high=0.01, size=(state_set, action_set))
    q_table = np.zeros((state_set, action_set))
    pi = np.full(state_set, 4)

    q_net = QNET()
    q_target_net = QNET()
    q_target_net.load_state_dict(q_net.state_dict())
    optimizer = torch.optim.SGD(q_net.parameters(), lr=0.001)

    batch_size = 100
    update_step = 10

    env.reset()
    episode = env.explore_by_epsilon_greedy_policy(pi, 5000, 1, False, 0.01)
    dataloader = get_data_loader(episode, batch_size, env)

    loss_fn = torch.nn.MSELoss()

    step = 500
    for k in tqdm(range(step), desc="Training", unit="epochs"):
        for state_action, reward, next_state in dataloader:
            q_value = q_net(state_action)
            q_value_target = torch.empty((batch_size, 0))
            for action in range(action_set):
                s_a = torch.cat((next_state, torch.full((batch_size, 1), action)), dim=1)
                q_value_target = torch.cat((q_value_target, q_target_net(s_a)), dim=1)
            q_star = torch.max(q_value_target, dim=1, keepdim=True)[0]
            y_target_value = reward + gamma * q_star
            loss = loss_fn(q_value, y_target_value)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if k % update_step == 0 and k != 0:
                q_target_net.load_state_dict(q_net.state_dict())

        for state in range(state_set):
            x, y = env.state_to_pos(state)
            for action in range(action_set):
                q_table[state][action] = float(q_net(torch.tensor((x, y, action)).reshape(-1, 3)))
            pi[state] = np.argmax(q_table[state])
            v[state] = q_table[state][pi[state]]

    print(v.reshape((n, m)))
    print(pi.reshape((n, m)))

    env.reset()
    env.explore_by_deterministic_policy(pi, 20, True, 0.5)


if __name__ == "__main__":
    dqn()

"""
tensor([[0.0990, 0.0721, 0.0665, 0.0736, 0.0923],
        [0.0821, 0.0461, 0.0505, 0.0735, 0.1035],
        [0.1154, 0.1016, 0.0985, 0.0920, 0.0941],
        [0.0821, 0.0461, 0.0505, 0.0735, 0.1035],
        [0.0990, 0.0721, 0.0665, 0.0736, 0.0923],
        [0.0990, 0.0721, 0.0665, 0.0736, 0.0923],
        [0.1067, 0.0931, 0.0946, 0.1075, 0.1239],
        [0.0979, 0.0782, 0.0908, 0.1054, 0.1232],
        [0.0839, 0.0600, 0.0559, 0.0693, 0.0805],
        [0.1124, 0.0842, 0.0719, 0.0747, 0.0820],
        [0.1154, 0.1016, 0.0985, 0.0920, 0.0941],
        [0.1067, 0.0719, 0.0633, 0.0833, 0.1153],
        [0.1067, 0.0719, 0.0633, 0.0833, 0.1153],
        [0.1154, 0.1016, 0.0985, 0.0920, 0.0941],
        [0.0648, 0.0563, 0.0694, 0.0908, 0.1116],
        [0.1214, 0.0947, 0.0580, 0.0595, 0.0775],
        [0.1109, 0.0999, 0.0974, 0.1038, 0.1245],
        [0.1324, 0.0967, 0.0744, 0.0824, 0.1065],
        [0.1055, 0.0937, 0.0902, 0.0925, 0.1107],
        [0.1154, 0.1016, 0.0985, 0.0920, 0.0941],
        [0.1124, 0.0842, 0.0719, 0.0747, 0.0820],
        [0.0786, 0.0675, 0.0667, 0.0787, 0.0969],
        [0.0975, 0.0717, 0.0492, 0.0615, 0.0801],
        [0.0990, 0.0721, 0.0665, 0.0736, 0.0923],
        [0.1028, 0.0872, 0.0785, 0.0879, 0.1042],
        [0.0975, 0.0717, 0.0492, 0.0615, 0.0801],
        [0.1067, 0.0931, 0.0946, 0.1075, 0.1239],
        [0.1067, 0.0719, 0.0633, 0.0833, 0.1153],
        [0.1028, 0.0872, 0.0785, 0.0879, 0.1042],
        [0.0990, 0.0721, 0.0665, 0.0736, 0.0923],
        [0.0786, 0.0675, 0.0667, 0.0787, 0.0969],
        [0.0990, 0.0721, 0.0665, 0.0736, 0.0923],
        [0.1055, 0.0937, 0.0902, 0.0925, 0.1107],
        [0.1067, 0.0931, 0.0946, 0.1075, 0.1239],
        [0.1214, 0.0947, 0.0580, 0.0595, 0.0775],
        [0.1067, 0.0719, 0.0633, 0.0833, 0.1153],
        [0.0975, 0.0717, 0.0492, 0.0615, 0.0801],
        [0.1066, 0.0889, 0.0802, 0.0789, 0.0764],
        [0.0808, 0.0568, 0.0648, 0.0898, 0.1057],
        [0.0967, 0.0731, 0.0676, 0.0593, 0.0722],
        [0.0839, 0.0600, 0.0559, 0.0693, 0.0805],
        [0.0821, 0.0461, 0.0505, 0.0735, 0.1035],
        [0.1083, 0.0876, 0.0882, 0.0824, 0.0787],
        [0.1214, 0.0947, 0.0580, 0.0595, 0.0775],
        [0.1067, 0.0719, 0.0633, 0.0833, 0.1153],
        [0.1154, 0.1016, 0.0985, 0.0920, 0.0941],
        [0.1067, 0.0931, 0.0946, 0.1075, 0.1239],
        [0.0839, 0.0600, 0.0559, 0.0693, 0.0805],
        [0.1166, 0.0994, 0.0918, 0.0849, 0.0910],
        [0.0648, 0.0563, 0.0694, 0.0908, 0.1116],
        [0.1154, 0.1016, 0.0985, 0.0920, 0.0941],
        [0.1029, 0.0897, 0.0916, 0.1058, 0.1245],
        [0.1154, 0.1016, 0.0985, 0.0920, 0.0941],
        [0.0967, 0.0731, 0.0676, 0.0593, 0.0722],
        [0.1124, 0.0842, 0.0719, 0.0747, 0.0820],
        [0.0648, 0.0563, 0.0694, 0.0908, 0.1116],
        [0.1029, 0.0897, 0.0916, 0.1058, 0.1245],
        [0.0967, 0.0731, 0.0676, 0.0593, 0.0722],
        [0.1128, 0.0900, 0.0753, 0.0609, 0.0689],
        [0.1066, 0.0889, 0.0802, 0.0789, 0.0764],
        [0.0990, 0.0721, 0.0665, 0.0736, 0.0923],
        [0.1214, 0.0947, 0.0580, 0.0595, 0.0775],
        [0.0979, 0.0782, 0.0908, 0.1054, 0.1232],
        [0.1067, 0.0719, 0.0633, 0.0833, 0.1153],
        [0.0808, 0.0568, 0.0648, 0.0898, 0.1057],
        [0.1154, 0.1016, 0.0985, 0.0920, 0.0941],
        [0.1109, 0.0999, 0.0974, 0.1038, 0.1245],
        [0.0839, 0.0600, 0.0559, 0.0693, 0.0805],
        [0.0808, 0.0568, 0.0648, 0.0898, 0.1057],
        [0.1128, 0.0900, 0.0753, 0.0609, 0.0689],
        [0.0786, 0.0675, 0.0667, 0.0787, 0.0969],
        [0.1055, 0.0937, 0.0902, 0.0925, 0.1107],
        [0.1029, 0.0897, 0.0916, 0.1058, 0.1245],
        [0.1324, 0.0967, 0.0744, 0.0824, 0.1065],
        [0.0979, 0.0782, 0.0908, 0.1054, 0.1232],
        [0.1324, 0.0967, 0.0744, 0.0824, 0.1065],
        [0.0648, 0.0563, 0.0694, 0.0908, 0.1116],
        [0.1125, 0.0954, 0.0854, 0.0848, 0.0928],
        [0.1067, 0.0719, 0.0633, 0.0833, 0.1153],
        [0.0648, 0.0563, 0.0694, 0.0908, 0.1116],
        [0.1066, 0.0889, 0.0802, 0.0789, 0.0764],
        [0.1214, 0.0947, 0.0580, 0.0595, 0.0775],
        [0.0979, 0.0782, 0.0908, 0.1054, 0.1232],
        [0.0648, 0.0563, 0.0694, 0.0908, 0.1116],
        [0.1109, 0.0999, 0.0974, 0.1038, 0.1245],
        [0.1055, 0.0937, 0.0902, 0.0925, 0.1107],
        [0.0648, 0.0563, 0.0694, 0.0908, 0.1116],
        [0.1066, 0.0889, 0.0802, 0.0789, 0.0764],
        [0.0786, 0.0675, 0.0667, 0.0787, 0.0969],
        [0.1055, 0.0937, 0.0902, 0.0925, 0.1107],
        [0.1128, 0.0900, 0.0753, 0.0609, 0.0689],
        [0.1028, 0.0872, 0.0785, 0.0879, 0.1042],
        [0.1214, 0.0947, 0.0580, 0.0595, 0.0775],
        [0.0808, 0.0568, 0.0648, 0.0898, 0.1057],
        [0.1055, 0.0937, 0.0902, 0.0925, 0.1107],
        [0.0967, 0.0731, 0.0676, 0.0593, 0.0722],
        [0.1028, 0.0872, 0.0785, 0.0879, 0.1042],
        [0.0648, 0.0563, 0.0694, 0.0908, 0.1116],
        [0.0975, 0.0717, 0.0492, 0.0615, 0.0801],
        [0.1066, 0.0889, 0.0802, 0.0789, 0.0764]], grad_fn=<CatBackward0>)
tensor([[0.0990],
        [0.1035],
        [0.1154],
        [0.1035],
        [0.0990],
        [0.0990],
        [0.1239],
        [0.1232],
        [0.0839],
        [0.1124],
        [0.1154],
        [0.1153],
        [0.1153],
        [0.1154],
        [0.1116],
        [0.1214],
        [0.1245],
        [0.1324],
        [0.1107],
        [0.1154],
        [0.1124],
        [0.0969],
        [0.0975],
        [0.0990],
        [0.1042],
        [0.0975],
        [0.1239],
        [0.1153],
        [0.1042],
        [0.0990],
        [0.0969],
        [0.0990],
        [0.1107],
        [0.1239],
        [0.1214],
        [0.1153],
        [0.0975],
        [0.1066],
        [0.1057],
        [0.0967],
        [0.0839],
        [0.1035],
        [0.1083],
        [0.1214],
        [0.1153],
        [0.1154],
        [0.1239],
        [0.0839],
        [0.1166],
        [0.1116],
        [0.1154],
        [0.1245],
        [0.1154],
        [0.0967],
        [0.1124],
        [0.1116],
        [0.1245],
        [0.0967],
        [0.1128],
        [0.1066],
        [0.0990],
        [0.1214],
        [0.1232],
        [0.1153],
        [0.1057],
        [0.1154],
        [0.1245],
        [0.0839],
        [0.1057],
        [0.1128],
        [0.0969],
        [0.1107],
        [0.1245],
        [0.1324],
        [0.1232],
        [0.1324],
        [0.1116],
        [0.1125],
        [0.1153],
        [0.1116],
        [0.1066],
        [0.1214],
        [0.1232],
        [0.1116],
        [0.1245],
        [0.1107],
        [0.1116],
        [0.1066],
        [0.0969],
        [0.1107],
        [0.1128],
        [0.1042],
        [0.1214],
        [0.1057],
        [0.1107],
        [0.0967],
        [0.1042],
        [0.1116],
        [0.0975],
        [0.1066]], grad_fn=<MaxBackward0>)
"""
