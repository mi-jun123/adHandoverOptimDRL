import gymnasium as gym
import numpy as np
from gymnasium import spaces


class NetworkSwitchEnv(gym.Env):
    def __init__(self, state_params_config=None):
        """
        初始化网络切换环境。

        定义动作空间和观测空间，并初始化状态变量。

        参数:
        state_params_config (list, optional): 状态参数的配置列表，每个元素是一个元组 (param_name, low, high)，
            表示参数名、参数的最小值和最大值。默认配置为信噪比、速度和距离。
        """
        # 动作空间保持不变
        self.action_space = spaces.Discrete(3)

        # 处理状态空间配置
        # 检查传入的状态参数配置是否为 None
        if state_params_config is None:
            # 如果为 None，使用默认的状态参数配置
            # 这里定义了一个包含三个元组的列表，每个元组表示一个状态参数
            # 元组的第一个元素是参数名，第二个元素是参数的最小值，第三个元素是参数的最大值
            state_params_config = [
                ("snr", 0.0, 100.0),  # 信噪比（SNR），取值范围从 0.0 到 100.0
                ("speed", 0.0, 100.0),  # 速度，取值范围从 0.0 到 100.0
                ("distance", 0.0, 1000.0),  # 距离，取值范围从 0.0 到 1000.0
                ("sinr", 0.0, 100.0),  # 距离，取值范围从 0.0 到 100.0
                ("current_network", 0, 1)  # 当前网络：0 - 自组网，1 - 5G
            ]

        # 从 state_params_config 中提取所有状态参数的名称，存储在 self.state_params 列表中
        # 使用列表推导式遍历 state_params_config 中的每个元组，取元组的第一个元素（即参数名）
        self.state_params = [param[0] for param in state_params_config]

        # 从 state_params_config 中提取所有状态参数的最小值，存储在 self.state_lows 数组中
        # 使用列表推导式遍历 state_params_config 中的每个元组，取元组的第二个元素（即参数的最小值）
        # 然后使用 np.array 将列表转换为 numpy 数组，并指定数据类型为 np.float32
        self.state_lows = np.array([param[1] for param in state_params_config], dtype=np.float32)

        # 从 state_params_config 中提取所有状态参数的最大值，存储在 self.state_highs 数组中
        # 使用列表推导式遍历 state_params_config 中的每个元组，取元组的第三个元素（即参数的最大值）
        # 然后使用 np.array 将列表转换为 numpy 数组，并指定数据类型为 np.float32
        self.state_highs = np.array([param[2] for param in state_params_config], dtype=np.float32)

        # 状态空间定义
        self.observation_space = spaces.Box(
            low=self.state_lows,
            high=self.state_highs,
            dtype=np.float32
        )

        # 初始化状态变量
        self.state = {}
        self.reset()

    def generate_external_params(self):

        return {
            "snr": np.random.uniform(0, 100),
            "speed": np.random.uniform(0, 100),
            "distance": np.random.uniform(0, 1000),
            "sinr": np.random.uniform(0, 100)
        }

    def step(self, action, external_params=None):
        """
        执行一个动作并更新环境状态。

        参数:
        action (int): 智能体选择的动作，取值为 0（自组网）、1（5G）或 2（保持不变）。

        返回:
        tuple: 包含以下元素的元组
            - observation (np.ndarray): 新的观测状态，包含所有状态参数。
            - reward (float): 执行动作后获得的奖励。
            - terminated (bool): 表示环境是否终止，这里始终为 False。
            - truncated (bool): 表示环境是否被截断，这里始终为 False。
            - info (dict): 包含额外信息的字典，如状态参数的变化、当前状态值。
        """
        old_state = self.state.copy()

        # 执行网络切换操作
        if action == 0:  # 切换到自组网
            self.state["current_network"] = 0
        elif action == 1:  # 切换到 5G
            self.state["current_network"] = 1
        elif action == 2:  # 保持不变
            pass

        # 根据外接参数更新状态
        if external_params is not None:
            for param, value in external_params.items():
                if param in self.state_params:
                    self.state[param] = np.clip(value, self.state_lows[self.state_params.index(param)],
                                                self.state_highs[self.state_params.index(param)])

            # 奖励函数优化，根据当前网络选择参数计算奖励
        reward = 0
        current_network = self.state["current_network"]
        if current_network == 0 and "snr" in self.state_params:
            # 自组网，使用 snr 计算奖励
            reward += (self.state["snr"] - 20) * 0.1  # 基准奖励
            reward += (self.state["snr"] - old_state.get("snr", 0)) * 0.5  # 变化奖励
        elif current_network == 1 and "sinr" in self.state_params:
            # 5G，使用 sinr 计算奖励
            reward += (self.state["sinr"] - 20) * 0.1
            reward += (self.state["sinr"] - old_state.get("sinr", 0)) * 0.5

        # 构建返回参数（根据最新 API 规范）
        observation = np.array([self.state[param] for param in self.state_params], dtype=np.float32)
        info = {
            f"{param}_change": self.state[param] - old_state.get(param, 0) for param in self.state_params
        }

        info.update(self.state)

        return (
            observation,
            float(reward),
            False,  # terminated
            False,  # truncated
            info
        )

    def reset(self, seed=None, options=None):
        """
        重置环境状态。

        参数:
        seed (int, optional): 随机种子，用于固定随机数生成。
        options (dict, optional): 额外的重置选项，这里未使用。

        返回:
        tuple: 包含以下元素的元组
            - observation (np.ndarray): 重置后的观测状态，包含所有状态参数。
            - info (dict): 包含额外信息的字典，如重置消息。
        """
        # 1. 调用父类的 reset 方法（可选，但推荐）
        super().reset(seed=seed)

        # 2. 固定随机种子（关键步骤）
        if seed is not None:
            np.random.seed(seed)  # 固定 numpy 的随机种子
            # 如需固定其他库的种子（如 random），可添加：
            # import random
            # random.seed(seed)

        # 初始化状态变量
        for i, param in enumerate(self.state_params):
            if param == "current_network":
                # 随机初始化当前网络
                self.state[param] = np.random.randint(0, 2)
            else:
                self.state[param] = np.random.uniform(self.state_lows[i], self.state_highs[i])

        observation = np.array([self.state[param] for param in self.state_params], dtype=np.float32)
        return (
            observation,
            {"message": "Environment reset"}  # info
        )

    def render(self, mode="human"):
        """
        渲染环境。

        参数:
        mode (str): 渲染模式，支持 "human" 和 "rgb_array"。

        返回:
        np.ndarray or None: 如果模式为 "rgb_array"，返回一个空的 RGB 数组；如果模式为 "human"，打印环境信息并返回 None。
        """
        # 实现基础的人类可读渲染
        if mode == "human":
            status_str = "Network Status | "
            for param in self.state_params:
                if param == "current_network":
                    network_name = "自组网" if self.state[param] == 0 else "5G"
                    status_str += f"{param}: {network_name} | "
                else:
                    status_str += f"{param}: {self.state[param]:.1f} | "
            print(status_str.rstrip(" | "))
            return None
        # 支持 rgb_array 模式返回空数组（符合 API 规范）
        elif mode == "rgb_array":
            return np.empty((64, 64, 3), dtype=np.uint8)
        else:
            raise NotImplementedError(f"Render mode {mode} not supported")

    def close(self):
        """
        关闭环境并清理资源。

        这里没有需要清理的资源，所以方法为空。
        """
        # 清理资源（如果需要）
        pass