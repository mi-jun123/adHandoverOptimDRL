import gymnasium as gym
import numpy as np
from gymnasium import spaces
import time

class NetworkSwitchEnv(gym.Env):
    def __init__(self, state_params_config=None):
        """
        初始化网络切换环境。

        定义动作空间和观测空间，并初始化状态变量。

        参数:
        state_params_config (list, optional): 状态参数的配置列表，每个元素是一个元组 (param_name, low, high)，
            表示参数名、参数的最小值和最大值。默认配置为信噪比、速度、距离、信干噪比和当前网络。
        """
        # 动作空间：0 - 自组网，1 - 5G，2 - 保持不变
        self.action_space = spaces.Discrete(3)

        # 处理状态空间配置
        if state_params_config is None:
            state_params_config = [
                ("snr", 0.0, 100.0),  # 信噪比（SNR），取值范围从 0.0 到 100.0
                ("speed", 0.0, 100.0),  # 速度，取值范围从 0.0 到 100.0
                ("distance", 0.0, 1000.0),  # 距离，取值范围从 0.0 到 1000.0
                ("sinr", 0.0, 100.0),  # 信干噪比（SINR），取值范围从 0.0 到 100.0
                ("current_network", 0, 1)  # 当前网络：0 - 自组网，1 - 5G
            ]
        self.state_params = [param[0] for param in state_params_config]
        self.state_lows = np.array([param[1] for param in state_params_config], dtype=np.float32)
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

        self.consecutive_low_sinr = 0  # 连续低 SINR 计数器
        self.success_threshold = 1 # 成功奖励阈值



        """
        执行一个动作并更新环境状态。

        参数:
        action (int): 智能体选择的动作，取值为 0（自组网）、1（5G）或 2（保持不变）。
        external_params (dict, optional): 外接的网络参数，键为参数名，值为参数值。

        返回:
        tuple: 包含以下元素的元组
            - observation (np.ndarray): 新的观测状态，包含所有状态参数。
            - reward (float): 执行动作后获得的奖励。
            - terminated (bool): 表示环境是否终止，这里始终为 False。
            - truncated (bool): 表示环境是否被截断，这里始终为 False。
            - info (dict): 包含额外信息的字典，如状态参数的变化、当前状态值。
        """

    def step(self, action, external_params=None):
        old_state = self.state.copy()

        # 处理外部参数
        if external_params is None:
            external_params = {
                "snr": 14,
                "speed": 0,
                "distance": 0,
                "sinr": 0
            }

        # 更新状态
        for param, value in external_params.items():
            if param in self.state_params:
                self.state[param] = np.clip(value, self.state_lows[self.state_params.index(param)],
                                            self.state_highs[self.state_params.index(param)])

        # 执行动作
        if action == 0:
            self.state["current_network"] = 0
        elif action == 1:
            self.state["current_network"] = 1
        elif action == 2:
            pass  # 保持不变

        # 计算奖励（优化逻辑）
        reward = 5
        current_network = self.state["current_network"]
        sinr = self.state["sinr"]
        snr = self.state["snr"]

        # 网络切换奖励
        # 定义一个常量来表示切换的惩罚，可根据实际情况调整
        SWITCH_PENALTY = 0

        if current_network == 1 and sinr <= 1:
            if action == 0:
                # 当 5G 信号差时切换到自组网，给予奖励，但减去切换惩罚
                reward += 10 - SWITCH_PENALTY
            else:
                # 未切换到自组网给予负向奖励
                reward -= 5
        elif current_network == 0:
            # 自组网奖励：基于 SNR 稳定性
            # 基准奖励
            reward += (snr - 15) * 0.5
            # 变化奖励
            reward += (snr - old_state.get("snr", 0)) * 0.5

            # 检查是否从 5G 切换到自组网
            if old_state.get("current_network", 0) == 1 and action == 0:
                # 如果是从 5G 切换到自组网，减去切换惩罚
                reward -= SWITCH_PENALTY

                # 对比自组网和 5G 的信号质量
            old_sinr = old_state.get("sinr", 0)
            if old_sinr > 1 and snr < old_sinr:
                # 如果自组网信号质量比之前的 5G 略差，给予一定的负向奖励
                reward -= 1

        elif current_network == 1 and sinr > 1:
            # 5G 奖励：基于 SINR 稳定性
            # 基准奖励
            reward += (sinr - 20) * 0.5
            # 变化奖励
            reward += (sinr - old_state.get("sinr", 0)) * 0.5

            # 检查是否从自组网切换到 5G
            if old_state.get("current_network", 0) == 0 and action == 1:
                # 如果是从自组网切换到 5G，减去切换惩罚
                reward -= SWITCH_PENALTY
                # 惩罚频繁切换
        if action != 2:
            reward -= 2  # 每次切换扣 2 分
        # 终止条件判断
        terminated = False
        truncated = False

        # 失败条件：连续 3 步 5G SINR ≤ 1 且未切换到自组网
        if self.state["current_network"] == 1 and self.state["sinr"] <= 1 and action != 0:
            self.consecutive_low_sinr += 1
            if self.consecutive_low_sinr >= 3:
                terminated = True
                reward -= 50  # 失败惩罚
        else:
            self.consecutive_low_sinr = 0  # 重置计数器

        # 成功条件：单步奖励 ≥5
        if reward >= self.success_threshold:
            truncated = True
            reward += 50  # 成功奖励
        print(f"reward ：{reward} ")
        return (
            np.array([self.state[param] for param in self.state_params], dtype=np.float32),
            float(reward),
            terminated,
            truncated,
            {"state": self.state}
        )


    def reset(self, seed=None, options=None):
        # 固定随机种子
        if seed is not None:
            np.random.seed(seed)

        # 初始化状态（增强随机性）
        self.state = {
            "snr": np.random.uniform(10, 20),  # 随机初始化 SNR（10-20）
            "speed": np.random.uniform(0, 100),
            "distance": np.random.uniform(0, 1000),
            "sinr": np.random.uniform(0, 100),  # 随机初始化 SINR
            "current_network": np.random.randint(0, 1)  # 随机初始网络
        }

        # 限制参数范围
        for param in self.state_params:
            self.state[param] = np.clip(
                self.state[param],
                self.state_lows[self.state_params.index(param)],
                self.state_highs[self.state_params.index(param)]
            )

        observation = np.array([self.state[param] for param in self.state_params], dtype=np.float32)
        self.consecutive_low_sinr = 0  # 重置计数器
        return observation, {"message": "Environment reset"}

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




