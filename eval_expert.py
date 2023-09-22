from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from gym.envs.box2d import CarRacing
from gym.wrappers.frame_stack import FrameStack
from gym.wrappers import TimeLimit
from gym.wrappers import RecordVideo

from env import EpisodeReward, TransformAction, TransformImage
from policy import Policy
from PIL import Image
import numpy as np

eval_path = Path('eval_expert/')
eval_path.mkdir(exist_ok=True)
env = CarRacing(render_mode="rgb_array", domain_randomize=False)
env = FrameStack(env, 4)
env = TransformImage(env)
env = TimeLimit(env, 1000)
env = TransformAction(env)

actor_critic = Policy(
    env.observation_space.shape,
    env.action_space,
    activation=None
)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
actor_critic = actor_critic.to(device)

saved_variables = torch.load('ckpt/model_45.pt', map_location='cuda')
actor_critic.load_state_dict(saved_variables)

images = []
obs, _ = env.reset()
done = False
step = 0
ep_reward = 0
image_array = env.render()
while not done:
    if step % 12 == 0:
        images.append(image_array)
    obs = torch.from_numpy(obs).float()
    obs = obs.unsqueeze(0).to(device)
    with torch.no_grad():
        _, action, _ = actor_critic.act(obs, deterministic=True)
    action = action[0].cpu().numpy()
    obs, reward, done, truncated, info = env.step(action)
    ep_reward += reward
    done |= truncated
    step += 1
    image_array = env.render()

print(ep_reward)
print('step: ', step)
mosaico = np.zeros([8 * image_array.shape[0], 8 * image_array.shape[1], image_array.shape[2]], dtype=np.uint8)
for i in range(8):
    for j in range(8):
        i_image = min(len(images) - 1, i * 8 + j)
        mosaico[i * image_array.shape[0]:(i + 1) * image_array.shape[0], j * image_array.shape[1]:(j + 1) * image_array.shape[1], :] = images[i_image]

im = Image.fromarray(mosaico)
im.save('mosaico.png')