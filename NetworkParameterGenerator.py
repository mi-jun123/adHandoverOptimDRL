import numpy as np

class ExternalParameterGenerator:
    def __init__(self, drop_prob=0.1, min_drop_duration=3, max_drop_duration=8):
        """
        初始化外部参数生成器。

        :param drop_prob: 掉线概率
        :param min_drop_duration: 最小掉线持续步数
        :param max_drop_duration: 最大掉线持续步数
        """
        self.drop_prob = drop_prob
        self.min_drop_duration = min_drop_duration
        self.max_drop_duration = max_drop_duration
        self.is_dropped = False
        self.drop_remaining = 0

    def generate_external_params(self):
        """
        生成包含 SNR、速度、距离和 SINR 的外部参数字典，实时考虑掉线情况。

        :return: 外部参数字典
        """
        if self.is_dropped:
            sinr = 0
            self.drop_remaining -= 1
            if self.drop_remaining == 0:
                self.is_dropped = False
        else:
            if np.random.rand() < self.drop_prob:
                self.is_dropped = True
                self.drop_remaining = np.random.randint(self.min_drop_duration, self.max_drop_duration + 1)
                sinr = 0
            else:
                random_value = np.random.rand()
                if random_value < 0.1:
                    sinr = 0
                elif random_value < 0.2:
                    sinr = np.random.choice([0, 1])
                else:
                    sinr = 15 + np.random.normal(0, 1)

        snr = 14+np.random.normal(0, 1)
        speed = np.random.uniform(0, 100)
        distance = np.random.uniform(0, 1000)
        return {
            "snr": snr,
            "speed": speed,
            "distance": distance,
            "sinr": sinr
        }