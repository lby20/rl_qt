import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal
from tianshou.policy import DDPGPolicy
from tianshou.utils.net.common import Net
from tianshou.data import Collector, ReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.trainer import offpolicy_trainer

# 确保您的自定义环境已经导入
from custom_environment import FuturesTradingEnv

# 定义Actor网络
class Actor(nn.Module):
    def __init__(self, state_shape, action_shape):
        super().__init__()
        self.net = Net(state_shape, hidden_sizes=[128, 128], device='cuda')
        self.rnn = nn.GRU(128, 64, batch_first=True)
        self.mu = nn.Linear(64, np.prod(action_shape))
        self.sigma = nn.Linear(64, np.prod(action_shape))
        self.action_shape = action_shape

    def forward(self, s, state=None, info={}):
        s, hidden = self.net(s, state)
        s, h = self.rnn(s.unsqueeze(0))
        mu = torch.tanh(self.mu(s)).squeeze(0)
        sigma = torch.softplus(self.sigma(s)).squeeze(0)
        return Normal(mu, sigma), h

# 定义Critic网络
class Critic(nn.Module):
    def __init__(self, state_shape, action_shape):
        super().__init__()
        self.net = Net(state_shape, action_shape, hidden_sizes=[128, 128], concat=True, device='cuda')
        self.rnn = nn.GRU(128, 64, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, s, a=None, state=None, info={}):
        s, hidden = self.net(s, a, state)
        s, h = self.rnn(s.unsqueeze(0))
        value = self.fc(s).squeeze(0)
        return value, h

# 定义环境和相关参数
env = DummyVectorEnv([lambda: FuturesTradingEnv(win_len=30, data_file_path="./data/IC_2015to2018.csv", furtures="IC") for _ in range(10)])
state_shape = env.observation_space.shape or env.observation_space.n
action_shape = env.action_space.shape or env.action_space.n
train_envs = DummyVectorEnv([lambda: FuturesTradingEnv(win_len=30, data_file_path="./data/IC_2015to2018.csv", furtures="IC") for _ in range(10)])
test_envs = DummyVectorEnv([lambda: FuturesTradingEnv(win_len=30, data_file_path="./data/IC_2015to2018.csv", furtures="IC") for _ in range(10)])

# 创建网络实例
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
actor = Actor(state_shape, action_shape).to(device)
critic = Critic(state_shape, action_shape).to(device)

# 创建策略
policy = DDPGPolicy(actor, actor, critic, nn.MSELoss(reduction='sum'), nn.MSELoss(reduction='sum'),
                    action_range=[float(env.action_space.low.min()), float(env.action_space.high.max())],
                    actor_optim=torch.optim.Adam(actor.parameters(), lr=0.001),
                    critic_optim=torch.optim.Adam(critic.parameters(), lr=0.001))

# 创建数据收集器
train_collector = Collector(policy, train_envs, ReplayBuffer(20000))
test_collector = Collector(policy, test_envs)

# 训练
result = offpolicy_trainer(policy, train_collector, test_collector, max_epoch=10, step_per_epoch=1000,
                           step_per_collect=10, update_per_step=0.1, batch_size=64, train_fn=lambda e: [None],
                           test_fn=lambda e: [None], stop_fn=lambda mean_rewards: mean_rewards >= env.spec.reward_threshold)

# 测试
test_collector.collect(n_episode=10, render=0.05)
