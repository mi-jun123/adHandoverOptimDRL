def evaluate_policy(env, agent, param_generator, turns = 3, seed=None):
    total_scores = 0

    for j in range(turns):
        print(f"正在评估第 {j + 1} 个回合...")
        step_count = 0
        try:
            s, info = env.reset(seed=seed)
            done = False
            while not done:
                # Take deterministic actions at test time
                step_count += 1
                print(f"第 {j + 1} 个回合，第 {step_count} 步")
                a = agent.select_action(s, deterministic=True)
                external_params = param_generator  # 生成外部参数
                s_next, r, dw, tr, info = env.step(a, external_params)  # 传入外部参数
                done = (dw or tr)

                total_scores += r
                s = s_next
        except Exception as e:
            print(f"评估第 {j + 1} 回合时出现异常: {e}")
    return total_scores / turns


#You can just ignore this funciton. Is not related to the RL.
def str2bool(v):
    '''transfer str to bool for argparse'''
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'True','true','TRUE', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'False','false','FALSE', 'f', 'n', '0'):
        return False
    else:
        print('Wrong Input.')
        raise