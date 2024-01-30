import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches


class Info:
    def __init__(self, state, action, reward, next_state):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state


class Environment:
    def __init__(self, graph: np.ndarray, reward: np.ndarray):
        # game
        self.graph = graph  # graph is a matrix. 0: empty, 1: forbidden, 2: target
        self.reward = reward  # reward[0] = 0, reward[1] = -1, reward[2] = 1
        self.agent_pos = (0, 0)
        self.trajectory = []
        # action = {0: up, 1: right, 2: down, 3: left, 4: stop}
        self.dx = [-1, 0, 1, 0, 0]
        self.dy = [0, 1, 0, -1, 0]

        # render
        self.agent = None
        self.size = len(graph)
        self.fig = None
        self.ax = None

    def reset(self, state=0):
        self.agent_pos = self.state_to_pos(state)
        self.trajectory.clear()

        # render
        plt.close()
        self.fig = plt.figure(figsize=(10, 10), dpi=self.size * 20)  # 设置画布，参数分别为画布大小和分辨率
        self.ax = plt.gca()  # 获取坐标轴对象，对图像的操作都是通过 ax 来实现的
        self.ax.set_aspect('equal', adjustable='box')  # 保持坐标轴刻度相等
        self.ax.xaxis.tick_top()  # 将 x 轴刻度放置在坐标轴的顶部
        self.ax.yaxis.tick_left()  # 将 y 轴刻度放置在坐标轴的左侧
        self.ax.invert_yaxis()
        self.ax.xaxis.set_ticks(range(0, self.size + 1))
        self.ax.yaxis.set_ticks(range(0, self.size + 1))
        self.ax.grid(True, linestyle="-", color="gray", linewidth="1", axis='both')
        self.ax.tick_params(bottom=False, left=False, right=False, top=False, labelbottom=False, labelleft=False,
                            labeltop=False)  # 将所有的刻度线都被设置为不可见，因为画的是网格，不需要刻度
        self.show_edge_id()
        self.show_state_id()
        self.colour_state()
        self.trajectory = []
        self.agent = patches.Arrow(0.5, 0.5, 0.4, 0, color='red', width=0.5)  # 一个箭头对象，代指 agent
        # self.ax.add_patch(self.agent)  # 通过 ax 将箭头加到画布上

    def next_state(self, state: int, action: int):
        pos = self.state_to_pos(state)
        action = int(action)
        next_pos = (pos[0] + self.dx[action], pos[1] + self.dy[action])
        if 0 <= next_pos[0] < self.size and 0 <= next_pos[1] < self.size:
            return next_pos[0] * self.size + next_pos[1]
        else:
            return state

    def acton_reward(self, state: int, action: int):
        pos = self.state_to_pos(state)
        action = int(action)
        next_pos = (pos[0] + self.dx[action], pos[1] + self.dy[action])
        if 0 <= next_pos[0] < self.size and 0 <= next_pos[1] < self.size:
            return self.pos_reward(next_pos)
        else:
            return self.reward[-1]

    def explore_by_deterministic_policy(self, pi: np.ndarray, step=0, show=False, speed=0.5):
        for _ in range(step):
            self.take_action(int(pi[self.pos_to_state(self.agent_pos)]), show, speed)
        return self.trajectory

    def get_epsilon_greedy_action(self, pi: np.ndarray, epsilon=0.0):
        probabilities = np.full(5, epsilon / 5)
        probabilities[pi[self.pos_to_state(self.agent_pos)]] += 1 - epsilon
        return np.random.choice(5, p=probabilities)

    def explore_by_epsilon_greedy_policy(self, pi: np.ndarray, step=0, epsilon=0.0, show=False, speed=0.5):
        for _ in range(step):
            self.take_action(self.get_epsilon_greedy_action(pi, epsilon), show, speed)
        return self.trajectory

    def take_action(self, action=4, show=False, show_speed=0.5):
        next_pos = (self.agent_pos[0] + self.dx[action], self.agent_pos[1] + self.dy[action])
        if 0 <= next_pos[0] < self.size and 0 <= next_pos[1] < self.size:
            next_pos, self.agent_pos = self.agent_pos, next_pos
            self.draw_random_line(self.agent_pos, next_pos)
            self.trajectory.append(Info(self.pos_to_state(next_pos), action, self.pos_reward(self.agent_pos), self.pos_to_state(self.agent_pos)))
        else:
            self.trajectory.append(Info(self.pos_to_state(self.agent_pos), action, self.pos_reward(self.agent_pos), self.pos_to_state(self.agent_pos)))

        if show:
            self.show(show_speed)

        return self.trajectory[-1]

    def state_to_pos(self, state):
        return state // self.size, state % self.size

    def pos_to_state(self, pos):
        return int(pos[0] * self.size + pos[1])

    # return the reward of state[pos]
    def pos_reward(self, pos):
        return self.reward[self.graph[pos[0]][pos[1]]]

    def show(self, show_speed=0.5):
        plt.pause(show_speed)

    def show_edge_id(self):
        for y in range(self.size):
            self.write_word(pos=(-0.6, y), word=str(y + 1), size_discount=0.8)
            self.write_word(pos=(y, -0.6), word=str(y + 1), size_discount=0.8)

    def show_state_id(self):
        index = 0
        for y in range(self.size):
            self.write_word(pos=(-0.6, y), word=str(y + 1), size_discount=0.8)
            self.write_word(pos=(y, -0.6), word=str(y + 1), size_discount=0.8)
            for x in range(self.size):
                self.write_word(pos=(x, y), word="s" + str(index), size_discount=0.65)
                index += 1

    def colour_state(self):
        for i in range(self.size):
            for j in range(self.size):
                if self.graph[i][j] == 1:
                    self.fill_block(pos=(j, i))
                if self.graph[i][j] == 2:
                    self.fill_block(pos=(j, i), color='darkturquoise')

    def write_word(self, pos, word, color='black', y_offset=0, size_discount=1.0) -> None:
        self.ax.text(pos[0] + 0.5, pos[1] + 0.5 + y_offset, word, size=size_discount * (30 - 2 * self.size),
                     ha='center', va='center', color=color)

    def fill_block(self, pos, color='#EDB120', width=1.0, height=1.0) -> patches.RegularPolygon:
        return self.ax.add_patch(patches.Rectangle((pos[0], pos[1]), width=1.0, height=1.0,
                                                   facecolor=color, fill=True, alpha=0.90))

    def draw_random_line(self, pos1, pos2) -> None:
        offset1 = np.random.uniform(low=-0.05, high=0.05, size=1)
        offset2 = np.random.uniform(low=-0.05, high=0.05, size=1)
        x = [pos1[1] + 0.5, pos2[1] + 0.5]
        y = [pos1[0] + 0.5, pos2[0] + 0.5]
        if pos1[1] == pos2[1]:
            x = [x[1] + offset1, x[0] + offset2]
        else:
            y = [y[1] + offset1, y[0] + offset2]
        self.ax.plot(x, y, color='g', scalex=False, scaley=False)



if __name__ == "__main__":
    graph = [
        [0, 1],
        [0, 2]
    ]
    reward = [0, -1, 1]
    env = Environment(graph, reward)
    env.reset()
    # traj = env.equal_random_explore(3, True, 1)
    # cnt = [0, 0, 0, 0, 0]
    # for i in range(len(traj)):
    #     cnt[traj[i][1]] += 1
    # for x in cnt:
    #     print(x)

    while True:
        env.draw_random_line((0, 0), (0, 1))
        env.show(0.5)

