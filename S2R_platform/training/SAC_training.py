import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from simulator.training_env import USVSimulator
from simulator.path_manager import generate_straight_path, PathManager

# ========== 模型定义 ==========
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
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.zeros_(m.bias)

    def forward(self, state):
        state = self.fc_state(state)
        mu = self.fc_mu(state)
        std = self.fc_std(state).clamp(min=1e-6)
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
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.zeros_(m.bias)

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

# ========== 训练环境 ==========
waypoints = generate_straight_path(length=1000, angle_deg=30, interval=1)
path_manager = PathManager(waypoints)

# 全部关闭
wind_config = {'method': None, 'speed': 0.0, 'direction': 0.0}
wave_config = {'method': None}
current_config = {'method': None}


env = USVSimulator(
    path_manager=path_manager,
    ship_model='mariner',
    wind_config=wind_config,
    wave_config=wave_config,
    current_config=current_config,
    dt=0.1
)

# ========== 训练数据与辅助函数 ==========
reward_history = []
datas = []

fig, ax = plt.subplots(figsize=(10, 5))
alpha = torch.tensor(np.log(0.01), requires_grad=True)

def get_action(state):
    state = torch.FloatTensor(state).reshape(1, 3)
    action, _ = model_action(state)
    return action.item()

def update_data():
    state = env.reset()
    over, total_reward, step_count = False, 0, 0
    while not over and step_count < 1000:
        action = get_action(state)
        next_state, reward, over, _ = env.step([action])
        datas.append((state, action, reward, next_state, over))
        state = next_state
        step_count += 1
        total_reward += reward
    while len(datas) > 50000:
        datas.pop(0)
    return total_reward

def get_sample():
    samples = random.sample(datas, 64)
    state = torch.FloatTensor([i[0] for i in samples]).reshape(-1, 3)
    action = torch.FloatTensor([i[1] for i in samples]).reshape(-1, 1)
    reward = torch.FloatTensor([i[2] for i in samples]).reshape(-1, 1)
    next_state = torch.FloatTensor([i[3] for i in samples]).reshape(-1, 3)
    over = torch.LongTensor([i[4] for i in samples]).reshape(-1, 1)
    return state, action, reward, next_state, over

def get_target(next_state, reward, over):
    action, entropy = model_action(next_state)
    target1 = model_value_next1(next_state, action)
    target2 = model_value_next2(next_state, action)
    target = torch.min(target1, target2)
    target += alpha.exp() * entropy
    target *= 0.99 * (1 - over)
    target += reward
    return target

def get_loss_action(state):
    action, entropy = model_action(state)
    value1 = model_value1(state, action)
    value2 = model_value2(state, action)
    value = torch.min(value1, value2)
    return (-alpha.exp() * entropy - value).mean(), entropy

def soft_update(model, model_next):
    for p, p_next in zip(model.parameters(), model_next.parameters()):
        p_next.data.copy_(0.995 * p_next.data + 0.005 * p.data)

def plot_training_progress():
    ax.cla()
    ax.plot(reward_history, label='Total Reward')
    ax.set_title('Total Reward History')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Reward')
    ax.legend()
    plt.pause(0.001)

# ========== 训练主过程 ==========
def train():
    optimizer_action = torch.optim.Adam(model_action.parameters(), lr=1e-4)
    optimizer_value1 = torch.optim.Adam(model_value1.parameters(), lr=1e-3)
    optimizer_value2 = torch.optim.Adam(model_value2.parameters(), lr=1e-3)
    optimizer_alpha = torch.optim.Adam([alpha], lr=1e-5)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(300):
        total_reward = update_data()
        reward_history.append(total_reward)
        for _ in range(2000):
            state, action, reward, next_state, over = get_sample()
            reward = (reward + 8) / 8
            target = get_target(next_state, reward, over).detach()
            loss_value1 = loss_fn(model_value1(state, action), target)
            loss_value2 = loss_fn(model_value2(state, action), target)

            optimizer_value1.zero_grad(); loss_value1.backward(); torch.nn.utils.clip_grad_norm_(model_value1.parameters(), 2.0); optimizer_value1.step()
            optimizer_value2.zero_grad(); loss_value2.backward(); torch.nn.utils.clip_grad_norm_(model_value2.parameters(), 2.0); optimizer_value2.step()

            loss_action, entropy = get_loss_action(state)
            optimizer_action.zero_grad(); loss_action.backward(); torch.nn.utils.clip_grad_norm_(model_action.parameters(), 2.0); optimizer_action.step()

            loss_alpha = ((entropy + 1).detach() * alpha.exp()).mean()
            optimizer_alpha.zero_grad(); loss_alpha.backward(); torch.nn.utils.clip_grad_norm_([alpha], 2.0); optimizer_alpha.step()

            soft_update(model_value1, model_value_next1)
            soft_update(model_value2, model_value_next2)

        if epoch % 10 == 0:
            plot_training_progress()

    torch.save(model_action.state_dict(), 'model_action.pth')
    torch.save(model_value1.state_dict(), 'model_value1.pth')
    torch.save(model_value2.state_dict(), 'model_value2.pth')


# ========== 最终轨迹可视化 ==========
def plot_final_trajectory():
    state = env.reset()
    trajectory = []
    done = False
    while not done and len(trajectory) < 1000:
        action, _ = model_action(torch.FloatTensor(state).reshape(1, 3))
        next_state, _, done, _ = env.step([action.item()])
        full_state = env.get_full_state()
        trajectory.append([full_state[3], full_state[4]])
        state = next_state

    trajectory = np.array(trajectory)
    waypoints = np.array(env.waypoints)
    plt.figure(figsize=(10, 10))
    plt.plot(waypoints[:, 0], waypoints[:, 1], 'b--', label='Target Path')
    plt.plot(trajectory[:, 0], trajectory[:, 1], 'r', label='Agent Path')
    plt.title('Final Trajectory')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.legend()
    plt.show()


# 启动训练
train()
plot_training_progress()
model_action.load_state_dict(torch.load('model_action.pth'))
plot_final_trajectory()
