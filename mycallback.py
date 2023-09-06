import os
import sys
import gymnasium as gym
import numpy as np
import cv2
from pprint import pprint
from stable_baselines3.common.callbacks import BaseCallback

import matplotlib.pyplot as plt

from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results

from collections import deque


DEBUG = 'debug' in sys.argv

SAVE_IDENTIFIER = None
for arg in sys.argv:
    if arg.startswith("id="):
        split = arg.split('=')
        print(split)
        SAVE_IDENTIFIER = split[1]
if SAVE_IDENTIFIER == None:
    print("Please, specify the save identifier")
    sys.exit(1)

MODEL_TYPE = None
for arg in sys.argv:
    if arg.startswith("model="):
        split = arg.split('=')
        print(split)
        MODEL_TYPE = split[1].upper()
if MODEL_TYPE == None:
    print("Please, specify the baseline")
    sys.exit(1)


class MyCallback(BaseCallback):
    def __init__(self, env, verbose=0):
        super(MyCallback, self).__init__(verbose)
        self.env = env
        if DEBUG:
            cv2.namedWindow("vision", cv2.WINDOW_NORMAL) 
            cv2.resizeWindow("vision", 140*4+280, 80*2)
        self.accum_energy = 0
        self.accum_delta = 0


        self.check_freq = 1
        self.log_dir = "runs/"
        self.save_path = os.path.join(self.log_dir, "best_model")
        self.best_mean_reward = -np.inf

        self.queue = deque(maxlen=10)
        for i in range(10): self.queue.append(0.)
        self.reward = 0
        self.max_rollout_average_reward = 0
        self.rollout = -1

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
        self.reward = 0

    def _on_training_start(self) -> None:
        pass

    def _on_episode_start(self) -> None:
        print("episode")
        pass

    def _on_rollout_start(self) -> None:
        self.rollout += 1
        self.queue.append(self.reward)
        current_avg_rollout_reward = sum(self.queue)/len(self.queue)
        if current_avg_rollout_reward > self.max_rollout_average_reward:
            self.max_rollout_average_reward = current_avg_rollout_reward
            self.model.save(f"save_{MODEL_TYPE}_{SAVE_IDENTIFIER}_{str(self.rollout).zfill(10)}.pth")
        self.reward = 0
        pass

    def _on_step(self) -> bool:
        self.reward += self.env.reward

        # self.accum_delta += self.env.info['delta']
        # self.accum_energy += self.env.info['energy']

        # if DEBUG:
        #     # left_image = self.env.images["left_view"]
        #     # right_image = self.env.images["right_view"]
        #     image = self.env.images["view"]
        #     # cv2.imshow("vision", np.concatenate((left_image, right_image), axis=1))
        #     cv2.imshow("vision", image)
        #     k = cv2.waitKey(1)
        #     if k%255 == 113:
        #         sys.exit(0)
        #     if k%255 == 27:
        #         self.env.reset()


        return True

    def _on_rollout_end(self) -> None:
        pass

    def _on_training_end(self) -> None:
        pass





class MyImageSavingCallback(BaseCallback):
    def __init__(self, env, verbose=0):
        super(MyImageSavingCallback, self).__init__(verbose)
        self.env = env
        self.steps = 0
        if DEBUG:
            cv2.namedWindow("vision", cv2.WINDOW_NORMAL) 
            cv2.resizeWindow("vision", 140*4+280, 80*2)
        self.frame = -1
    def _on_training_start(self) -> None:
        pass

    def _on_rollout_start(self) -> None:
        pass

    def _on_step(self) -> bool:
        self.steps += 1


        if self.steps % 50 == 0:
            # print (self.steps)
            self.frame += 1
            if self.frame == 20_000:
                sys.exit(0)
            # left_image = self.env.images["left_view"]
            # right_image = self.env.images["right_view"]
            image = self.env.images["view"]

            # cv2.imwrite(f'left_images/left_image_{str(self.frame).zfill(6)}.png', left_image)
            # cv2.imwrite(f'right_images/right_image_{str(self.frame).zfill(6)}.png', right_image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.imwrite(f'images/image_{str(self.frame).zfill(6)}.png', image)

            # cv2.imshow("vision", np.concatenate((left_image, right_image), axis=1))
            # cv2.imshow("vision", image)
            # k = cv2.waitKey(1)
            # if k%255 == 113:
                # sys.exit(0)
            # if k%255 == 27:
                # self.env.reset()
        return True

    def _on_rollout_end(self) -> None:
        pass

    def _on_training_end(self) -> None:
        pass
