print(losses)

import matplotlib.pyplot as plt

video_path = "./video_bc"
env_video = CarRacing(render_mode="rgb_array", domain_randomize=False)
env_video = FrameStack(env_video, 4)
env_video = TransformImage(env_video)
env_video = TimeLimit(env_video, 1000)
env_video = TransformAction(env_video)
env_video = RecordVideo(env_video, video_path, episode_trigger=always_true)


plt.plot(losses_bc_samples, losses)

plt.plot(losses_bc_samples, entropies)

plt.plot(ep_rewards_samples, ep_rewards)

obs, _ = env_video.reset()
done = False
step = 0
while not done:
  obs = torch.from_numpy(obs).float()
  obs = obs.unsqueeze(0).to(device)
  with torch.no_grad():
      _, action, _ = bc_actor.act(obs, deterministic=True)
  action = action[0].cpu().numpy()
  obs, reward, done, truncated, info = env_video.step(action)
  done |= truncated
  step += 1

