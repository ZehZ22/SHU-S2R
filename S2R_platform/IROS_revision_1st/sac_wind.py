import random
import numpy as np
import torch
from matplotlib import pyplot as plt
from USVStateR import USVState, generate_path
from marinervalidation import mariner
from sim2 import marinerwind
from isherwood72 import isherwood72
from mariner2 import mariner2


# 生成路径
angle_deg = 30
length = 1000
interval = 1
waypoints = generate_path(angle_deg, length, interval)


# 定义风速.0
wind_velocity_ave = 0.0  # 平均风速 (以 m/s 为单位)
# 可以在这里选择 mariner 函数或其他自定义的动力学模型函数
env = USVState(
    waypoints=waypoints,
    current_index=1,
    x=np.array([0.0, 0.0, 0.0, 300.0, 0.0, np.radians(20), 0.0], dtype=np.float32),
    ui=0.0,
    model_func=marinerwind,

)

# 存储数据
reward_history = []
datas = []

# 初始化图形
fig, ax = plt.subplots(figsize=(10, 5))


# 权重初始化函数
def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)


class ModelAction(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_state = torch.nn.Sequential(
            torch.nn.Linear(3, 128),
            torch.nn.ReLU(),
        )
        self.fc_mu = torch.nn.Linear(128, 1)
        self.fc_std = torch.nn.Sequential(
            torch.nn.Linear(128, 1),
            torch.nn.Softplus(),
        )
        self.apply(init_weights)

    def forward(self, state):
        state = self.fc_state(state)
        mu = self.fc_mu(state)
        std = self.fc_std(state).clamp(min=1e-6)  # 防止 std 过小或为 NaN

        dist = torch.distributions.Normal(mu, std)
        sample = dist.rsample()
        action = torch.tanh(sample)
        log_prob = dist.log_prob(sample)
        entropy = log_prob - (1 - action.tanh() ** 2 + 1e-7).log()
        entropy = -entropy

        return action * 35, entropy



class ModelValue(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sequential = torch.nn.Sequential(
            torch.nn.Linear(4, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1),
        )
        self.apply(init_weights)

    def forward(self, state, action):
        state = torch.cat([state, action], dim=1)
        return self.sequential(state)


model_action = ModelAction()
model_value1 = ModelValue()
model_value2 = ModelValue()
model_value_next1 = ModelValue()
model_value_next2 = ModelValue()

model_value_next1.load_state_dict(model_value1.state_dict())
model_value_next2.load_state_dict(model_value2.state_dict())


def get_action(state):
    print(f"state: {state}")  # 添加打印语句
    state = torch.FloatTensor(state).reshape(1, 3)
    action, _ = model_action(state)
    return action.item()



def update_data():
    state = env.reset()
    over = False
    step_count = 0
    max_steps = 400
    total_reward = 0

    while not over and step_count < max_steps:
        action = get_action(state)
        next_state, reward, over, _ = env.step([action])
        datas.append((state, action, reward, next_state, over))
        state = next_state
        step_count += 1
        total_reward += reward

    if step_count >= max_steps:
        print("Reached maximum steps in update_data, possible issue with convergence or environment.")
        print(state)

    while len(datas) > 50000:
        datas.pop(0)

    return total_reward


def get_sample():
    if len(datas) < 128:
        print(f"Not enough samples in datas. Current length: {len(datas)}")
        return None, None, None, None, None
    samples = random.sample(datas, 128)
    state = torch.FloatTensor(np.array([i[0] for i in samples])).reshape(-1, 3)
    action = torch.FloatTensor(np.array([i[1] for i in samples])).reshape(-1, 1)
    reward = torch.FloatTensor(np.array([i[2] for i in samples])).reshape(-1, 1)
    next_state = torch.FloatTensor(np.array([i[3] for i in samples])).reshape(-1, 3)
    over = torch.LongTensor(np.array([i[4] for i in samples])).reshape(-1, 1)
    return state, action, reward, next_state, over


def get_value(state, action):
    return model_value1(state, action)


def get_target(next_state, reward, over):
    action, entropy = model_action(next_state)
    target1 = model_value_next1(next_state, action)
    target2 = model_value_next2(next_state, action)
    target = torch.min(target1, target2)
    target += alpha.exp() * entropy
    target *= 0.99
    target *= (1 - over)
    target += reward
    return target


def get_loss_action(state):
    action, entropy = model_action(state)
    value1 = model_value1(state, action)
    value2 = model_value2(state, action)
    value = torch.min(value1, value2)
    loss_action = -alpha.exp() * entropy
    loss_action -= value
    return loss_action.mean(), entropy


def soft_update(model, model_next):
    for param, param_next in zip(model.parameters(), model_next.parameters()):
        value = param_next.data * 0.995 + param.data * 0.005
        param_next.data.copy_(value)


import math

alpha = torch.tensor(math.log(0.01))
alpha.requires_grad = True


def plot_training_progress():
    ax.cla()
    if len(reward_history) > 0:
        ax.plot(reward_history, label='Total Reward')
        ax.set_title('Total Reward History')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Total Reward')
        ax.legend()
        plt.pause(0.001)


def train():
    optimizer_action = torch.optim.Adam(model_action.parameters(), lr=1e-4)
    optimizer_value1 = torch.optim.Adam(model_value1.parameters(), lr=1e-3)
    optimizer_value2 = torch.optim.Adam(model_value2.parameters(), lr=1e-3)
    optimizer_alpha = torch.optim.Adam([alpha], lr=1e-5)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(400):
        total_reward = update_data()  # 获取当前轮次的总奖励
        reward_history.append(total_reward)  # 将奖励追加到reward_history
        for i in range(1500):
            state, action, reward, next_state, over = get_sample()
            reward = (reward + 8) / 8
            target = get_target(next_state, reward, over).detach()
            value1 = model_value1(state, action)
            value2 = model_value2(state, action)
            loss_value1 = loss_fn(value1, target)
            loss_value2 = loss_fn(value2, target)
            optimizer_value1.zero_grad()
            loss_value1.backward()

            # 使用梯度裁剪
            torch.nn.utils.clip_grad_norm_(model_value1.parameters(), max_norm=2.0)

            optimizer_value1.step()
            optimizer_value2.zero_grad()
            loss_value2.backward()

            # 使用梯度裁剪
            torch.nn.utils.clip_grad_norm_(model_value2.parameters(), max_norm=2.0)

            optimizer_value2.step()
            loss_action, entropy = get_loss_action(state)
            optimizer_action.zero_grad()
            loss_action.backward()

            # 使用梯度裁剪
            torch.nn.utils.clip_grad_norm_(model_action.parameters(), max_norm=2.0)

            optimizer_action.step()
            loss_alpha = (entropy + 1).detach() * alpha.exp()
            loss_alpha = loss_alpha.mean()
            optimizer_alpha.zero_grad()
            loss_alpha.backward()

            # 使用梯度裁剪
            torch.nn.utils.clip_grad_norm_([alpha], max_norm=2.0)

            optimizer_alpha.step()
            soft_update(model_value1, model_value_next1)
            soft_update(model_value2, model_value_next2)
        if epoch % 10 == 0:
            plot_training_progress()

    torch.save(model_action.state_dict(), 'model_action.pth')
    torch.save(model_value1.state_dict(), 'model_value1.pth')
    torch.save(model_value2.state_dict(), 'model_value2.pth')


def plot_final_trajectory():
    state = env.reset()
    trajectory = []
    done = False
    step_count = 0

    while not done and step_count < 1000:
        action, _ = model_action(torch.FloatTensor(state).reshape(1, 3))
        next_state, _, done, _ = env.step([action.item()])
        full_state = env.get_full_state()
        trajectory.append([full_state[3], full_state[4]])
        state = next_state
        step_count += 1

    trajectory = np.array(trajectory)
    waypoints = np.array(env.waypoints)

    plt.figure(figsize=(10, 10))
    plt.plot(waypoints[:, 0], waypoints[:, 1], 'b--', label='Target Path')
    plt.plot(trajectory[:, 0], trajectory[:, 1], 'r', label='Agent Path')
    plt.title('Final Trajectory')
    plt.xlabel('X Coordinate (m)')
    plt.ylabel('Y Coordinate (m)')
    plt.legend()
    plt.grid(True)
    plt.show()


train()
plot_training_progress()

model_action.load_state_dict(torch.load('model_action.pth'))
plot_final_trajectory()
