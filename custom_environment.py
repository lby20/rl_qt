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

# def sample(df, )

class FuturesTradingEnv(gym.Env):
    def __init__(self, win_len, data_file_path, furtures):
        # furtures: ["IC", "IF"]
        super(FuturesTradingEnv, self).__init__()

        if furtures=="IC":
            self.contract_multiplier, self.min_security = 200, 0.2 # 每点200元, 最小变动价位0.2点 from http://www.cffex.com.cn/zz500/
        elif furtures=="IF":
            self.contract_multiplier, self.min_security = 300, 0.2 # 每点300元, 最小变动价位0.2点 from http://www.cffex.com.cn/hs300/
        # 观测历史价格的窗口宽度
        self.win_len = win_len
        
        # 初始化代码
        # 设置action_space和observation_space

        self.trans_fee=2.3*(10**(-5))
        self.slip = 0.2  # constant slippage
        
        # actions 有两个，long，short
        self.action_space = spaces.Discrete(2)
        self.market_data = preProcessData(data_file_path)
        self.len = len(self.market_data)
    def _init_account(self):
        principal = 5e5 # 本金
        return {
            "profit": 0,
            "margin": principal,
            "principal": principal,
            "a_pre": 0 # TODO: 在计算reward 时，需要a_pre,但是第一个a如何计算？
        }
    def step(self, action):
        # 处理一步交易逻辑
        # action：{long 0, short 1} -> 论文中为{-1, 1}
        if action==0:
            action = -1
        # 返回observation, reward, done, info
        # self.t = t
        done = (self.t == self.len)
        # o_{t}, a_t (根据0~t-1的历史做出决定，在t实现action) -> r_t (在t实现a_t 交易行为，产生的account profit)
        reward = (self.market_data[t]["close"] - self.market_data[t-1]["close"] - 2*self.slip)*self.account_info["a_pre"] \
                    - self.trans_fee*np.abs(self.account_info["a_pre"] - action)*self.market_data[t]["close"] # p_t^c - p_t-1^c .....
        self.account_info["a_pre"] = action
        self.t += 1
        observation = self.market_data[self.t - self.win_len: self.t] 
        
        info = self.account_info
        # 更新info
        return observation, reward, done, info

    def reset(self):
        # 模型输入需要观察wein_len 宽的历史数据，因此开票之后需要等一等
        self.t = self.win_len 
        self.market_obs = self.market_data[self.t - self.win_len: self.t] # [0, t-1]
        # 初始化账号信息
        self.account_info = self._get_info()
        # 返回初始观察值, account_info
        return self.market_obs, self.account_info

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