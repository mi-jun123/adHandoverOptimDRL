import torch
import gym
import numpy as np
from NetworkParameterGenerator import ExternalParameterGenerator
"""
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"可用的 GPU 设备: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("未检测到可用的 GPU 设备，将使用 CPU 运行。")
"""
generator = ExternalParameterGenerator()

# 设定测试次数
test_steps = 10

# 循环测试
for step in range(test_steps):
    # 生成外部参数
    external_params = generator.generate_external_params()

    # 打印每次生成的外部参数
    print(f"Step {step + 1}:")
    print(f"  SNR: {external_params['snr']}")
    print(f"  Speed: {external_params['speed']:.2f}")
    print(f"  Distance: {external_params['distance']:.2f}")
    print(f"  SINR: {external_params['sinr']:.2f}")
    print()