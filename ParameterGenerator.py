import numpy as np

class ExternalParameterGenerator:
    def __init__(self, rss_mean=-50, rss_std=2, sinr_mean=15, sinr_std=3, sinr_neg_prob=0.1, sinr_neg_min=-10,
                 sinr_neg_max=0, rtt_5g=40, rtt_ad_hoc=20, snr_ad_hoc=10, bandwidth_5g=100e6, bandwidth_ad_hoc=20e6,
                 c_5g=0.1, c_ad_hoc=0.9):
        """
        初始化外部参数生成器的参数
        :param rss_mean: RSS 的均值
        :param rss_std: RSS 的标准差
        :param sinr_mean: SINR 的均值
        :param sinr_std: SINR 的标准差
        :param sinr_neg_prob: SINR 产生负数的概率
        :param sinr_neg_min: SINR 负数的最小值
        :param sinr_neg_max: SINR 负数的最大值
        :param rtt_5g: 5G 的 RTT
        :param rtt_ad_hoc: 自组网的 RTT
        :param snr_ad_hoc: 自组网的 SNR
        :param bandwidth_5g: 5G 的带宽
        :param bandwidth_ad_hoc: 自组网的带宽
        :param c_5g: 5G 的成本
        :param c_ad_hoc: 自组网的成本
        """
        self.rss_mean = rss_mean
        self.rss_std = rss_std
        self.sinr_mean = sinr_mean
        self.sinr_std = sinr_std
        self.sinr_neg_prob = sinr_neg_prob
        self.sinr_neg_min = sinr_neg_min
        self.sinr_neg_max = sinr_neg_max
        self.rtt_5g = rtt_5g
        self.rtt_ad_hoc = rtt_ad_hoc
        self.snr_ad_hoc = snr_ad_hoc
        self.bandwidth_5g = bandwidth_5g
        self.bandwidth_ad_hoc = bandwidth_ad_hoc
        self.c_5g = c_5g
        self.c_ad_hoc = c_ad_hoc

    def _generate_rss(self):
        """
        生成 RSS 值
        :return: 生成的 RSS 值
        """
        return self.rss_mean + np.random.normal(0, self.rss_std)

    def _generate_sinr(self):
        """
        生成 SINR 值
        :return: 生成的 SINR 值
        """
        sinr = self.sinr_mean + np.random.normal(0, self.sinr_std)
        if np.random.random() < self.sinr_neg_prob:
            sinr = np.random.uniform(self.sinr_neg_min, self.sinr_neg_max)
        return sinr

    def external_params_generator(self):
        """
        生成外部网络参数
        :return: 包含网络参数的字典
        """
        # 5G 参数
        rss_5g = self._generate_rss()
        rtt_5g = self.rtt_5g
        sinr_5g = self._generate_sinr()
        bandwidth_5g = self.bandwidth_5g
        c_5g = self.c_5g

        # 自组网参数
        rss_ad_hoc = self._generate_rss()
        rtt_ad_hoc = self.rtt_ad_hoc
        snr_ad_hoc = self.snr_ad_hoc
        bandwidth_ad_hoc = self.bandwidth_ad_hoc
        c_ad_hoc = self.c_ad_hoc

        params = {
            "rss_5g": rss_5g,
            "rtt_5g": rtt_5g,
            "sinr_5g": sinr_5g,
            "bandwidth_5g": bandwidth_5g,
            "c_5g": c_5g,
            "rss_ad_hoc": rss_ad_hoc,
            "rtt_ad_hoc": rtt_ad_hoc,
            "snr_ad_hoc": snr_ad_hoc,
            "bandwidth_ad_hoc": bandwidth_ad_hoc,
            "c_ad_hoc": c_ad_hoc
        }
        return params