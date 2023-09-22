import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


expert_ep_rewards = [[480.61210191081807] * 40, [552.8662420381982] * 40, [490.45095541400235] * 40]
results_array = np.array(expert_ep_rewards)
results_mean = results_array.mean(axis=0)
results_std = results_array.std(axis=0)
plt.plot(range(0, 200, 5), results_mean, label='Expert Evaluation')
plt.fill_between(range(0, 200, 5), results_mean + results_std, results_mean - results_std, alpha=0.5)

for results_type, label in zip(['', 'deterministic_'], ['BC with Stochastic Expert', 'BC with Deterministic Expert']):
    ep_rewards_list = []
    for i_run in range(3):
        ep_rewards = pd.read_csv(f'{results_type}{i_run}/ep_rewards.csv')
        ep_rewards_list.append(ep_rewards['ep_rewards'])

    results_array = np.array(ep_rewards_list)
    results_mean = results_array.mean(axis=0)
    results_std = results_array.std(axis=0)

    plt.plot(range(0, 200, 5), results_mean, label=label)
    plt.fill_between(range(0, 200, 5), results_mean + results_std, results_mean - results_std, alpha=0.5)

plt.legend(loc='best', shadow=True, fontsize='medium')
plt.xlabel('Training epoch')
plt.ylabel('Episode Reward')
plt.savefig('ep_rewards.png')
plt.clf()
