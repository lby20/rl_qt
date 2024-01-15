import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pandas as pd

def randomDate(start, end, format):
    stime = datetime.strptime(start, format)
    etime = datetime.strptime(end, format)
    prop = random.random()
    ptime = stime + prop * (etime - stime)
    return ptime.strftime(format)

def preProcessData(file):
    """
    对读取raw data, 并进行初步整理
    """
    df=pd.read_csv(file,parse_dates=True,index_col=0)
    df=df.reset_index()
    df.rename(columns={df.columns[0]: "datetime"}, inplace=True)
    return df

def episode(df, day):

    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # 筛选特定日期的数据
    specific_date = '2019-12-30'
    filtered_df = df[df['datetime'].dt.date == pd.to_datetime(specific_date).date()]
    # 显示结果
    return filtered_df

    
class FuturesTradingEnv(gym.Env):
    def __init__(self, data_fn):
        super(FuturesTradingEnv, self).__init__()
        # 初始化代码
        # 设置action_space和observation_space

        self.transFee=2.3*(10**(-5))
        self.slip = 0.2  # constant slippage
        
        # actions 有两个，long，short
        self.action_space = spaces.Discrete(2)
        self.observation_space = ...

    def step(self, action):
        # 处理一步交易逻辑
        # 返回observation, reward, done, info
        return observation, reward, done, info

    def reset(self):
        # 重置环境状态
        # 返回初始观察值
        return initial_observation

    def render(self, mode='human'):
        # 可选：渲染环境状态
        pass

    def close(self):
        # 可选：清理工作
        pass

#     def 
# # 使用环境的示例
# env = FuturesTradingEnv(data_fn="your_data.csv", ...)
# observation = env.reset()
# done = False
# while not done:
#     action = env.action_space.sample()  # 或者您的代理生成的行动
#     observation, reward, done, info = env.step(action)
#     env.render()
# env.close()
df = preProcessData("./data/IC_2015to2018.csv")
print(type(df.iloc[0, 0]))
df = episode(df, "2019-12-30")
print(df)