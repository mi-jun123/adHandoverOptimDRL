import argparse
import os
import shutil
from datetime import datetime
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm  # 导入 tqdm 库

from DQN import DQN_agent
from myenv import NetworkSwitchEnv
from utils import evaluate_policy, str2bool
from gym import spaces

'''Hyperparameter Setting'''
parser = argparse.ArgumentParser()
parser.add_argument('--dvc', type=str, default='cuda', help='running device: cuda or cpu')
parser.add_argument('--EnvIdex', type=int, default=0, help='CP-v1, LLd-v2')
parser.add_argument('--write', type=str2bool, default=True, help='Use SummaryWriter to record the training')
parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')
parser.add_argument('--Loadmodel', type=str2bool, default=False, help='Load pretrained model or Not')
parser.add_argument('--ModelIdex', type=int, default=100, help='which model to load')

parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--Max_train_steps', type=int, default=int(1e2), help='Max training steps')
parser.add_argument('--save_interval', type=int, default=int(50e3), help='Model saving interval, in steps.')
parser.add_argument('--eval_interval', type=int, default=int(2e3), help='Model evaluating interval, in steps.')
parser.add_argument('--random_steps', type=int, default=int(3e3), help='steps for random policy to explore')
parser.add_argument('--update_every', type=int, default=50, help='training frequency')

parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
parser.add_argument('--net_width', type=int, default=200, help='Hidden net width')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=256, help='lenth of sliced trajectory')
parser.add_argument('--exp_noise', type=float, default=0.2, help='explore noise')
parser.add_argument('--noise_decay', type=float, default=0.99, help='decay rate of explore noise')
parser.add_argument('--Double', type=str2bool, default=True, help='Whether to use Double Q-learning')
parser.add_argument('--Duel', type=str2bool, default=True, help='Whether to use Duel networks')
opt = parser.parse_args()
opt.dvc = torch.device(opt.dvc)  # from str to torch.device
print(opt)

# 模拟参数生成函数
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

# 修改 generate_external_params 函数
def generate_external_params(sinr_sequence, step_index):
    sinr = sinr_sequence[step_index]
    snr = 14
    speed = np.random.uniform(0, 100)
    distance = np.random.uniform(0, 1000)
    return {
        "snr": snr,
        "speed": speed,
        "distance": distance,
        "sinr": sinr
    }

def main():
    EnvName = 'NetworkSwitchEnv'
    BriefEnvName = 'NSE'
    env = NetworkSwitchEnv()
    eval_env = NetworkSwitchEnv()
    opt.state_dim = env.observation_space.shape[0]
    opt.action_dim = env.action_space.n
    opt.max_e_steps = None

    # 算法设置...
    if opt.Duel:
        algo_name = 'Duel'
    else:
        algo_name = ''
    if opt.Double:
        algo_name += 'DDQN'
    else:
        algo_name += 'DQN'

    # 随机种子设置...
    env_seed = opt.seed
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("Random Seed: {}".format(opt.seed))

    print('Algorithm:', algo_name, '  Env:', BriefEnvName[opt.EnvIdex], '  state_dim:', opt.state_dim,
          '  action_dim:', opt.action_dim, '  Random Seed:', opt.seed, '  max_e_steps:', opt.max_e_steps, '\n')

    if opt.write:
        timenow = str(datetime.now())[0:-10]
        timenow = ' ' + timenow[0:13] + '_' + timenow[-2::]
        writepath = 'runs/{}-{}_S{}_'.format(algo_name, BriefEnvName[opt.EnvIdex], opt.seed) + timenow
        if os.path.exists(writepath):
            shutil.rmtree(writepath)
        writer = SummaryWriter(log_dir=writepath)

    # 构建模型和回放缓冲区...
    if not os.path.exists('model'):
        os.mkdir('model')
    agent = DQN_agent(**vars(opt))
    if opt.Loadmodel:
        agent.load(algo_name, BriefEnvName[opt.EnvIdex], opt.ModelIdex)

    total_steps = 0
    # 生成 sinr 序列，长度为 Max_train_steps
    sinr_sequence = generate_sinr_sequence(opt.Max_train_steps)

    # 使用 tqdm 为训练循环添加进度条
    with tqdm(total=opt.Max_train_steps, desc="Training Progress", unit="step") as pbar:
        while total_steps < opt.Max_train_steps:
            s, info = env.reset(seed=env_seed)
            env_seed += 1
            done = False

            while not done:
                # 生成实时的外部参数
                external_params = generate_external_params(sinr_sequence, total_steps)

                # 交互和训练逻辑...
                if total_steps < opt.random_steps:
                    a = env.action_space.sample()
                else:
                    a = agent.select_action(s, deterministic=False)
                s_next, r, dw, tr, info = env.step(a, external_params)  # 传入外部参数
                done = (dw or tr)

                agent.replay_buffer.add(s, a, r, s_next, dw)
                s = s_next

                if total_steps >= opt.random_steps and total_steps % opt.update_every == 0:
                    for j in range(opt.update_every):
                        agent.train()

                if total_steps % 1000 == 0:
                    agent.exp_noise *= opt.noise_decay

                if total_steps % opt.eval_interval == 0:
                    score = evaluate_policy(eval_env, agent, turns=3)
                    if opt.write:
                        writer.add_scalar('ep_r', score, global_step=total_steps)
                        writer.add_scalar('noise', agent.exp_noise, global_step=total_steps)
                    print('EnvName:', BriefEnvName[opt.EnvIdex], 'seed:', opt.seed, 'steps: {}k'.format(int(total_steps / 1000)), 'score:', int(score))

                # 输出当前时间步信息
                print(f"Current step: {total_steps}, Reward: {r}, Noise: {agent.exp_noise:.4f}")

                # 更新进度条信息
                pbar.set_postfix({"Reward": f"{r:.2f}", "Noise": f"{agent.exp_noise:.4f}"})

                total_steps += 1
                pbar.update(1)  # 更新进度条

                if total_steps % opt.save_interval == 0:
                    agent.save(algo_name, BriefEnvName[opt.EnvIdex], int(total_steps / 1000))

    if opt.write:
        writer.close()
    env.close()
    eval_env.close()


if __name__ == '__main__':
    main()