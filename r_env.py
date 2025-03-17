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

            if state_params_config is None:
                state_params_config = [
                    # 自组网参数
                    ("rss_ad_hoc", -120.0, -10.0),  # 调整取值范围
                    ("rtt_ad_hoc", 0.0, 100.0),
                    ("bitrate_ad_hoc", 0.0, 1e9),
                    ("c_ad_hoc", 0, 1),
                    # 5G 参数
                    ("rtt_5g", 0.0, 100.0),
                    ("rss_5g", -120.0, -10.0),  # 调整取值范围
                    ("bitrate_5g", 0.0, 1e9),
                    ("c_5g", 0, 1),
                    ("current_network", 0, 1)
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
        """
        def calculate_path_loss(distance, frequency):
           
            计算自由空间路径损耗
            :param distance: 节点间距离，单位：km
            :param frequency: 信号频率，单位：MHz
            :return: 路径损耗，单位：dB
            
            return 20 * np.log10(distance) + 20 * np.log10(frequency) + 32.45
        """
       #path_loss = calculate_path_loss(distance, frequency)
        #receive_power = transmit_power - path_loss
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

    def calculate_UB(self,eta1, b, b_min=2,b0=0.075,B0=0.49,theta=97.8):
        """
        根据公式计算 U_B(b)

        参数:
        eta1 (float): 公式中的参数 η₁
        Dist_b (float): Dist(b) 的值
        b (float): 输入变量 b
        b_min (float): 阈值 b_min

        返回:
        float: U_B(b) 的计算结果
        """
        # 单位阶跃函数判断
        if b >= b_min:
            step = 1
        else:
            step = 0

        # 计算对数部分
        denominator = b - b0
        if denominator == 0:
            raise ValueError("分母不能为零，rj - r0 不能等于 0")
        dist_b=theta/denominator+B0
        log_part = math.log10((255 ** 2) / dist_b)

        # 计算最终结果
        UB = eta1 * 10 * log_part * step
        return UB

    def calculate_UD(self,eta2, d, d_min=150):
        """
        根据公式 U_D(d) = η₂·u(d_min - d) 计算结果，其中 u 为单位阶跃函数

        :param eta2: 公式中的参数 η₂（浮点数）
        :param d: 输入变量 d（浮点数）
        :param d_min: 阈值 d_min（浮点数）
        :return: U_D(d) 的计算结果
        """
        # 实现单位阶跃函数逻辑
        if d_min - d >= 0:
            unit_step = 1
        else:
            unit_step = 0

        UD=eta2 * unit_step
        return UD

    def calculate_URSS(self,eta3, rss, rssmin=-70):
        """
        根据公式 U_RSS(rss) = η₃·rss·u(rss - rss_min) 计算结果
        考虑到rss多半为负值，应该将实际的
        :param eta3: 公式中的参数 η₃（浮点数）
        :param rss: 接收信号强度 rss（浮点数）
        :param rssmin: 阈值 rss_min（浮点数）
        :return: U_RSS(rss) 的计算结果
        """
        # 处理单位阶跃函数
        if rss >= rssmin:
            unit_step = 1
        else:
            unit_step = 0

        URSS=-eta3 * rss * unit_step
        return URSS

    def calculate_UC(self,network_type,c0,c1):
        """
        根据公式 \( U_C(c) =
        \begin{cases}
        1 & \text{if WiFi network} \\
        0.1 & \text{otherwise}
        \end{cases} \) 计算结果。

        :param network_type: 网络类型
        :return: \(U_C(c)\) 的计算结果
        """
        if network_type== 0:
            UC=1.0
        elif network_type== 1:
            UC=c1/c0
        return  UC
    def normalize_utilities(self, utilities):
        """
        对效用值进行归一化处理
        :param utilities: 包含 UC, URSS, UD, UB 的列表
        :return: 归一化后的效用值列表
        """
        min_val = min(utilities)
        max_val = max(utilities)
        if max_val - min_val == 0:
            return [0] * len(utilities)
        return [(u - min_val) / (max_val - min_val) for u in utilities]

    def calculate_G(self, omega_b, omega_d, omega_Rs, omega_c, Ub, Ud, U_Rs, Uc):
        """
        计算 G(x, a_t) 函数
        :param omega_b: 比特率效用权重
        :param omega_d: 时延效用权重
        :param omega_Rs: 信号强度效用权重
        :param omega_c: 成本效用权重
        :param Ub: 比特率效用值
        :param Ud: 时延效用值
        :param U_Rs: 信号强度效用值
        :param Uc: 成本效用值
        :return: G(x, a_t) 的计算结果
        """
        utilities = [Ub, Ud, U_Rs, Uc]
        normalized_utilities = self.normalize_utilities(utilities)
        Ub_norm, Ud_norm, U_Rs_norm, Uc_norm = normalized_utilities
        return omega_b * Ub_norm + omega_d * Ud_norm + omega_Rs * U_Rs_norm + omega_c * Uc_norm


    def calculate_Q(self, alpha, G_value, a_t):
        """
        计算 Q(x, a_t) 函数
        :param alpha: 权重参数 (0.5, 1]
        :param G_value: G(x, a_t) 的计算结果
        :param a_t: 当前动作
        :param a_prev: 上一时刻动作
        :return: Q(x, a_t) 的计算结果
        """
        indicator_diff = 0 if a_t == 2 else 1
        indicator_same = 1 if a_t == 2 else 0

        return (1 - alpha) * G_value * indicator_diff + alpha * G_value * indicator_same

    def calculate_reward(self, state, action):
        # 从状态中提取所需参数
        if state["current_network"] == 0:
            rss = state["rss_ad_hoc"]
            rtt= state["rtt_ad_hoc"]
            bitrate = state["bitrate_ad_hoc"]
            c0 = state["c_ad_hoc"]
        else:
            rss = state["rss_5g"]
            rtt= state["rtt_5g"]
            bitrate = state["bitrate_5g"]
            c1 = state["c_5g"]

        # 计算各个效用值
        eta1 = 0.5
        eta2 = 0.6
        eta3 = 0.7
        Ub = self.calculate_UB(eta1, bitrate)
        Ud = self.calculate_UD(eta2, rtt)  # 假设时延为20，需要根据实际情况修改
        U_Rs = self.calculate_URSS(eta3, rss)
        Uc = self.calculate_UC(state["current_network"],c0,c1)

        # 计算 G 函数
        omega_b = 0.25
        omega_d = 0.25
        omega_Rs = 0.25
        omega_c = 0.25
        G_value = self.calculate_G(omega_b, omega_d, omega_Rs, omega_c, Ub, Ud, U_Rs, Uc)

        # 计算 Q 函数
        alpha = 0.8
        Q_value = self.calculate_Q(alpha, G_value, action)

        return Q_value


    def step(self, action, external_params=None):
        old_state = self.state.copy()
        # 更新状态
        if external_params is not None:
            for param, value in external_params.items():
                if param in self.state_params:
                    self.state[param] = np.clip(value, self.state_lows[self.state_params.index(param)],
                                                self.state_highs[self.state_params.index(param)])

            # 检查是否有计算比特率所需的参数
            if (
                    'snr_5g' in external_params and 'distance' in external_params and 'bandwidth_5g' in external_params and 'frequency' in external_params) or \
                    (
                            'sinr_ad_hoc' in external_params and 'distance' in external_params and 'bandwidth_ad_hoc' in external_params and 'frequency' in external_params):
                current_network = self.state["current_network"]
                if current_network == 0:  # 自组网
                    sinr = external_params['sinr_ad_hoc']
                    distance = external_params['distance']
                    bandwidth = external_params['bandwidth_ad_hoc']
                    frequency = external_params['frequency']
                    bitrate = self.calculate_bitrate(snr=sinr, distance=distance, bandwidth=bandwidth,
                                                     frequency=frequency)
                    self.state["bitrate_ad_hoc"] = bitrate
                elif current_network == 1:  # 5G
                    snr = external_params['snr_5g']
                    distance = external_params['distance']
                    bandwidth = external_params['bandwidth_5g']
                    frequency = external_params['frequency']
                    bitrate = self.calculate_bitrate(snr=snr, distance=distance, bandwidth=bandwidth,
                                                     frequency=frequency)
                    self.state["bitrate_5g"] = bitrate
                    # 更新 rss 和 rtt
                current_network = self.state["current_network"]
                if current_network == 0:  # 自组网
                    if 'rss_ad_hoc' in external_params:
                            self.state["rss_ad_hoc"] = external_params['rss_ad_hoc']
                    if 'rtt_ad_hoc' in external_params:
                            self.state["rtt_ad_hoc"] = external_params['rtt_ad_hoc']
                elif current_network == 1:  # 5G
                    if 'rss_5g' in external_params:
                            self.state["rss_5g"] = external_params['rss_5g']
                    if 'rtt_5g' in external_params:
                            self.state["rtt_5g"] = external_params['rtt_5g']
        # 执行动作
        prev_action = self.state["current_network"]
        if action == 0:
            self.state["current_network"] = 0
        elif action == 1:
            self.state["current_network"] = 1
        elif action == 2:
            pass  # 保持不变

        # 计算奖励
        reward = self.calculate_reward(self.state, action, prev_action)
        print(f"reward ：{reward} ")

        observation = np.array([self.state[param] for param in self.state_params], dtype=np.float32)
        info = {
            f"{param}_change": self.state[param] - old_state.get(param, 0) for param in self.state_params
        }
        info.update(self.state)

        terminated = False
        truncated = False
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

        for param_name, low, high in self.state_params_config:
            # 随机初始化状态变量
            if param_name == "current_network":
                self.state[param_name] = np.random.randint(low, high + 1)
            else:
                self.state[param_name] = np.random.uniform(low, high)

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




