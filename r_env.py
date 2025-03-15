import gymnasium as gym
import numpy as np
from gymnasium import spaces
import time

class NetworkSwitchEnv(gym.Env):

    def calculate_path_loss(distance, frequency):
        """
        计算自由空间路径损耗
        :param distance: 距离，单位：km
        :param frequency: 信号频率，单位：MHz
        :return: 路径损耗，单位：dB
        """
        return 20 * np.log10(self.distance) + 20 * np.log10(frequency) + 32.45



    def calculate_abr(sinr, bandwidth, efficiency=0.8):
        """
        计算可用比特率（ABR）
        :param sinr: 信号与干扰加噪声比（SINR）
        :param bandwidth: 信道带宽，单位：Hz
        :param efficiency: 效率因子，通常小于 1
        :return: 可用比特率，单位：bps
        """
        return efficiency * bandwidth * np.log2(1 + sinr)

    def calculate_abr(snr, bandwidth, efficiency=0.8):
        """
        计算可用比特率（ABR）
        :param snr: 信噪比（SNR）
        :param bandwidth: 信道带宽，单位：Hz
        :param efficiency: 效率因子，通常小于 1
        :return: 可用比特率，单位：bps
        """
        return efficiency * bandwidth * np.log2(1 + snr)

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
            # 定义默认的状态参数配置，区分自组网和 5G
            state_params_config = [
                # 自组网参数
                ("speed_ad_hoc", 0.0, 100.0),
                ("distance_ad_hoc", 0.0, 1000.0),
                ("sinr_ad_hoc", 0.0, 100.0),
                ("bandwidth_ad_hoc", 1e6, 100e6),
                ("bitrate_ad_hoc", 0.0, 1e9),
                # 5G 参数
                ("snr_5g", 0.0, 100.0),
                ("speed_5g", 0.0, 100.0),
                ("distance_5g", 0.0, 1000.0),
                ("bandwidth_5g", 1e6, 100e6),
                ("bitrate_5g", 0.0, 1e9),
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
        # 计算归一化参数
        self.state_mins = self.state_lows
        self.state_maxs = self.state_highs
        self.reset()

        sinr_min, sinr_max = self.state_mins[self.state_params.index("sinr")], self.state_maxs[self.state_params.index("sinr")]
        snr_min, snr_max = self.state_mins[self.state_params.index("snr")], self.state_maxs[self.state_params.index("snr")]
        self.norm_sinr_threshold_1 = (1 - sinr_min) / (sinr_max - sinr_min)
        self.norm_sinr_threshold_20 = (20 - sinr_min) / (sinr_max - sinr_min)
        self.norm_snr_threshold_15 = (15 - snr_min) / (snr_max - snr_min)
        self.success_threshold = 0
        self.consecutive_low_sinr = 0

    def calculate_bitrate(self, snr, distance, bandwidth, frequency, transmit_power=20, noise_power=-95,
                          efficiency=0.8):
        """
        计算可用比特率（ABR）
        :param snr: 信噪比（SNR）
        :param distance: 节点间距离，单位：km
        :param bandwidth: 信道带宽，单位：Hz
        :param frequency: 信号频率，单位：MHz
        :param transmit_power: 发射功率，单位：dBm
        :param noise_power: 噪声功率，单位：dBm
        :param efficiency: 效率因子，通常小于 1
        :return: 可用比特率，单位：bps
        """

        def calculate_path_loss(distance, frequency):
            """
            计算自由空间路径损耗
            :param distance: 节点间距离，单位：km
            :param frequency: 信号频率，单位：MHz
            :return: 路径损耗，单位：dB
            """
            return 20 * np.log10(distance) + 20 * np.log10(frequency) + 32.45

        path_loss = calculate_path_loss(distance, frequency)
        receive_power = transmit_power - path_loss
        snr_linear = 10 ** (snr / 10)
        abr = efficiency * bandwidth * np.log2(1 + snr_linear)
        return abr

    def normalize_state(self, state):
            """
            对状态进行归一化处理，将状态值缩放到 [0, 1] 范围。
            """
            if isinstance(state, dict):
                normalized_state = {}
                for param in self.state_params:
                    value = state[param]
                    min_val = self.state_mins[self.state_params.index(param)]
                    max_val = self.state_maxs[self.state_params.index(param)]
                    normalized_value = (value - min_val) / (max_val - min_val)
                    normalized_state[param] = normalized_value
                return normalized_state
            elif isinstance(state, np.ndarray):
                normalized_state = (state - self.state_mins) / (self.state_maxs - self.state_mins)
                return normalized_state
            else:
                raise ValueError("Unsupported state type. Expected dict or numpy.ndarray.")




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
            # 对状态进行归一化

        self.state = self.normalize_state(self.state)

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

        if current_network == 1 and sinr <= self.norm_sinr_threshold_1:
            if action == 0:
                # 当 5G 信号差时切换到自组网，给予奖励，但减去切换惩罚
                reward += 10 - SWITCH_PENALTY
            else:
                # 未切换到自组网给予负向奖励
                reward -= 5
        elif current_network == 0:
            # 自组网奖励：基于 SNR 稳定性
            # 基准奖励
            reward += (snr - self.norm_snr_threshold_15) * 0.5
            # 变化奖励
            old_snr = old_state.get("snr", 0)
            reward += (snr - old_snr) * 0.5

            # 检查是否从 5G 切换到自组网
            if old_state.get("current_network", 0) == 1 and action == 0:
                # 如果是从 5G 切换到自组网，减去切换惩罚
                reward -= SWITCH_PENALTY

            # 对比自组网和 5G 的信号质量
            old_sinr = old_state.get("sinr", 0)
            if old_sinr > self.norm_sinr_threshold_1 and snr < old_sinr:
                # 如果自组网信号质量比之前的 5G 略差，给予一定的负向奖励
                reward -= 1

        elif current_network == 1 and sinr > self.norm_sinr_threshold_1:
            # 5G 奖励：基于 SINR 稳定性
            # 基准奖励
            reward += (sinr - self.norm_sinr_threshold_20) * 0.5
            # 变化奖励
            old_sinr = old_state.get("sinr", 0)
            reward += (sinr - old_sinr) * 0.5

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
        if self.state["current_network"] == 1 and self.state["sinr"] <= self.norm_sinr_threshold_1 and action != 0:
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

        observation = np.array([self.state[param] for param in self.state_params], dtype=np.float32)
        info = {
            f"{param}_change": self.state[param] - old_state.get(param, 0) for param in self.state_params
        }
        info.update(self.state)

        return (
            observation,
            float(reward),
            terminated,
            truncated,
            info
        )

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        for i, param in enumerate(self.state_params):
            if param == "current_network":
                self.state[param] = np.random.randint(0, 2)
            elif param == "snr":
                self.state[param] = 14
            elif param == "sinr":
                self.state[param] = 15
            else:
                self.state[param] = np.random.uniform(self.state_lows[i], self.state_highs[i])

        observation = np.array([self.state[param] for param in self.state_params], dtype=np.float32)
        # 对状态进行归一化
        normalized_observation = self.normalize_state(observation)
        return (
            normalized_observation,
            {"message": "Environment reset"}
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




