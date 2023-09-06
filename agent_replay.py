from cmath import log
from subprocess import call
import sys

import numpy as np

from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise


import cv2
import survival
import tensorboard

from mycallback import MyCallback

DEBUG = 'debug' in sys.argv

if DEBUG:
    cv2.namedWindow("vision", cv2.WINDOW_NORMAL) 
    cv2.resizeWindow("vision", 140*4+280, 80*2)

env = survival.SurvivalEnv()


# The noise objects for DDPG
n_actions = env.action_space.shape[-1]
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))
model = DDPG("MultiInputPolicy", env, action_noise=action_noise, verbose=1, buffer_size=1000000, tensorboard_log="runs")
total_timesteps=300*100_000
log_interval=10
callback = MyCallback(env)
            
model = DDPG.load("ddpg")



obs = env.reset()
while True:
    print('episode')
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)

    # left_image = info['left_view']
    # right_image = info['right_view']
    # print(left_image.shape, right_image.shape)

    try:
        if DEBUG:
            cv2.imshow("vision", np.concatenate((left_image, right_image), axis=1))
    except Exception as e:
        print("error, e")
        pass

    if DEBUG:
        k = cv2.waitKey(1)
        if k%255 == 27:
            obs = env.reset()

    env.render()

