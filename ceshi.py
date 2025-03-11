import torch
import gym
import numpy as np

"""
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"可用的 GPU 设备: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("未检测到可用的 GPU 设备，将使用 CPU 运行。")
"""
def generate_sinr_sequence(num_steps, drop_prob=0.1, min_drop_duration=3, max_drop_duration=8):
        sinr_sequence = []
        is_dropped = False
        drop_remaining = 0

        for _ in range(num_steps):
            if is_dropped:
                sinr = 0
                drop_remaining -= 1
                if drop_remaining == 0:
                    is_dropped = False
            else:
                if np.random.rand() < drop_prob:
                    is_dropped = True
                    drop_remaining = np.random.randint(min_drop_duration, max_drop_duration + 1)
                    sinr = 0
                else:
                    random_value = np.random.rand()
                    if random_value < 0.6:
                        sinr = 0
                    elif random_value < 0.8:
                        sinr = np.random.choice([0, 1])
                    else:
                        sinr = 15 + np.random.normal(0, 1)

            sinr_sequence.append(sinr)

        return sinr_sequence


    # 功能测试
num_steps = 50
for _ in range(5):
        sinr_sequence = generate_sinr_sequence(num_steps)
        print("Generated SINR sequence:", sinr_sequence)
        # 检查是否存在连续的 0
        consecutive_zeros = 0
        for sinr in sinr_sequence:
            if sinr == 0:
                consecutive_zeros += 1
                if consecutive_zeros >= 3:  # 假设最小掉线持续时间为 3
                    print("Found a possible drop event with at least 3 consecutive zeros.")
                    break
            else:
                consecutive_zeros = 0
        else:
            print("No significant drop event found.")