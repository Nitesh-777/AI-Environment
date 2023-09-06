from cmath import log
from subprocess import call
import sys

import numpy as np

from stable_baselines3 import DDPG, TD3, PPO
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise


import cv2
import survival
import tensorboard

from mycallback import MyCallback, MyImageSavingCallback



MODEL_TYPE = None
for arg in sys.argv:
    if arg.startswith("model="):
        split = arg.split('=')
        print(split)
        MODEL_TYPE = split[1].upper()
if MODEL_TYPE == None:
    print("Please, specify the baseline")
    sys.exit(1)

SAVE_IDENTIFIER = None
for arg in sys.argv:
    if arg.startswith("id="):
        split = arg.split('=')
        print(split)
        SAVE_IDENTIFIER = split[1]
if SAVE_IDENTIFIER == None:
    print("Please, specify the save identifier")
    sys.exit(1)



env = survival.SurvivalEnv(render_cameras=True, render_view=True)


# The noise objects for DDPG
n_actions = env.action_space.shape[-1]
# action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))


if MODEL_TYPE == "DDPG":
    model = DDPG("MultiInputPolicy", env, action_noise=action_noise, verbose=1, buffer_size=1000000, tensorboard_log="runs")
elif MODEL_TYPE == "TD3":
    model = TD3("MultiInputPolicy", env, action_noise=action_noise, verbose=1, buffer_size=1000000, tensorboard_log="runs")
elif MODEL_TYPE == "PPO":
    model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log="runs")
else:
    print("MODEL_TYPE load")
    sys.exit(1)

total_timesteps=300*100_000
log_interval=1
callback = MyCallback(env)
# callback = MyImageSavingCallback(env)
model.learn(total_timesteps=total_timesteps, log_interval=log_interval, callback=callback)

# total_timesteps, callback = model._setup_learn(total_timesteps=total_timesteps, eval_env=None, callback=callback, eval_freq=-1,
#                                                n_eval_episodes=5, reset_num_timesteps=True, tb_log_name='run',)


# while model.num_timesteps < total_timesteps:
#     rollout = model.collect_rollouts(model.env, train_freq=model.train_freq, action_noise=model.action_noise, callback=callback,
#                                      learning_starts=model.learning_starts, replay_buffer=model.replay_buffer, log_interval=log_interval)
#     if rollout.continue_training is False:
#         break
#     if model.num_timesteps > 0 and model.num_timesteps > model.learning_starts:
#         # If no `gradient_steps` is specified,
#         # do as many gradients steps as steps performed during the rollout
#         gradient_steps = model.gradient_steps if model.gradient_steps >= 0 else rollout.episode_timesteps
#         # Special case when the user passes `gradient_steps=0`
#         if gradient_steps > 0:
#             model.train(batch_size=model.batch_size, gradient_steps=gradient_steps)
            

model.save(MODEL_TYPE)
env = model.get_env()

del model
model = DDPG.load(MODEL_TYPE)

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)

    # left_image = info['left_view']
    # right_image = info['right_view']
    
    # print(left_image.shape, right_image.shape)
    # try:
    #     if DEBUG:
    #         cv2.imshow("vision", np.concatenate((left_image, right_image), axis=1))
    # except Exception as e:
    #     print("error, e")
    #     pass

    # if DEBUG:
    #     k = cv2.waitKey(1)
    #     if k%255 == 27:
    #         obs = env.reset()

    env.render()

