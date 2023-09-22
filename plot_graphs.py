import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


loss_list = []
eval_loss_list = []
for i_run in range(2):
    loss = pd.read_csv(f'bc_{i_run}/loss.csv')
    eval_loss = pd.read_csv(f'{i_run}/eval_loss.csv')
    loss_list.append(loss['loss'])
    eval_loss_list.append(eval_loss['loss'])


loss_array = np.array(loss_list)
loss_mean = loss_array.mean(axis=0)
loss_std = loss_array.std(axis=0)

eval_loss_array = np.array(eval_loss_list)
eval_loss_mean = eval_loss_array.mean(axis=0)
eval_loss_std = eval_loss_array.std(axis=0)


plt.plot(range(0, 200), loss_mean, label='Training Data')
plt.fill_between(range(0, 200), loss_mean + loss_std, loss_mean - loss_std, alpha=0.5)

plt.plot(range(0, 200), eval_loss_mean, label='Validation Data')
plt.fill_between(range(0, 200), eval_loss_mean + eval_loss_std, eval_loss_mean - eval_loss_std, alpha=0.5)

plt.xlabel('Training epoch')
plt.ylabel('Loss function')
plt.legend(loc='best', shadow=True, fontsize='medium')

plt.savefig('loss.png')
plt.clf()

ep_rewards_list = []
for i_run in range(2):
    ep_rewards = pd.read_csv(f'bc_{i_run}/ep_rewards.csv')
    ep_rewards_list.append(ep_rewards['ep_rewards'])

results_array = np.array(ep_rewards_list)
results_mean = results_array.mean(axis=0)
results_std = results_array.std(axis=0)

plt.plot(range(0, 200, 5), results_mean)
plt.fill_between(range(0, 200, 5), results_mean + results_std, results_mean - results_std, alpha=0.5)

plt.xlabel('Training epoch')
plt.ylabel('Episode Reward')
plt.savefig('ep_rewards.png')
plt.clf()

# entropy = pd.read_csv('0/entropy.csv')

# plt.plot(entropy['bc_samples'], entropy['entropies'])
# plt.xlabel('Samples used in training')
# plt.ylabel('Entropy function')
# plt.savefig('entropy.png')
