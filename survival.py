from typing import Optional, Union

import time
import os
from pickletools import uint8
import sys
import numpy as np

from gymnasium import utils, spaces
from gymnasium.envs.mujoco import MujocoEnv
import gymnasium.envs.mujoco as mujoco
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

import numpy as np
import random

from vae import VAE
import matplotlib.pyplot as plt

import torch

np.random.seed(13)


WIDTH = 80
HEIGHT = 60
RENDER_FPS = 50
MAX_STEPS = 15000
FRAME_SKIP = 10
ROBOT_SPAWN_MAX_DISTANCE = 3
FOOD_SPAWN_MAX_DISTANCE = 10
FOOD_SPAWN_MIN_DISTANCE = 0.2
FOOD_DISTANCE_THRESHOLD = 0.6
FOOD_ITEMS = 10
NUMBER_OF_JOINTS = 12

INIT_HEALTH = 1000

DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}

DEBUG = 'debug' in sys.argv


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vae = VAE().to(device)
vae.load_state_dict(torch.load("vae.pth"))


class SurvivalEnv(MujocoEnv, utils.EzPickle):
    metadata = { "render_modes": [ "human", "rgb_array", "depth_array", ], "render_fps": RENDER_FPS, }

    def __init__(self, xml_file= os.getcwd() + "/assets/anto.xml", **kwargs):
        utils.EzPickle.__init__(**locals())

        self.height = HEIGHT
        self.width = WIDTH

        try:
            self.render_cameras = kwargs["render_cameras"]
        except:
            self.render_cameras = False
        try:
            self.render_view = kwargs["render_view"]
        except:
            self.render_view = False

        self.right_img, self.left_img, self.top_img = None, None, None

        self.observation_space = spaces.Dict({
            # "joint_position": spaces.Box(-1., 1., shape=(NUMBER_OF_JOINTS,), dtype=float),
            "joint_velocity": spaces.Box(-1., 1., shape=(2*NUMBER_OF_JOINTS,), dtype=float),
            "gyro":           spaces.Box(-1., 1., shape=(2*3,),   dtype=float),
            "accelerometer":  spaces.Box(-1., 1., shape=(2*3,),   dtype=float),
            "magnetometer":   spaces.Box(-1., 1., shape=(2*3,),   dtype=float),
            "z":              spaces.Box(-1., 1., shape=(128,), dtype=float),
        })
        # if self.render_cameras is True:
            # self.observation_space["view"] = spaces.Box(0, 255, shape=(HEIGHT, WIDTH, 3), dtype=float)
        
        self.ignore_contact_names = ['floor', 'gviewL1', 'gviewL']
        self.ignore_contact_ids = []

        # self.images = { "right_view": np.zeros((HEIGHT,WIDTH,3)), "left_view": np.zeros((HEIGHT,WIDTH,3))}
        self.images = { "view": np.zeros((HEIGHT,WIDTH,3))}

        MujocoEnv.__init__(self, xml_file, FRAME_SKIP, observation_space=self.observation_space, width=self.width, height=self.height)
        self.mujoco_renderer = MujocoRenderer(self.model, self.data)
        # try:
        #     self.render_mode = kwargs["render_mode"]
        # except:
        #     self.render_mode = "none"

        self.reset_model()


    def _set_action_space(self):
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        return self.action_space

    @property
    def done(self):
        return self._done

    def respawn_food(self, food_idx, avoid_locations=None, threshold=0.0):
        def get_random_position():
            WEIRD = False
            if WEIRD:
                distance = random.random()+FOOD_SPAWN_MAX_DISTANCE/4
                angle = (random.random()-0.5)*2.
                angle = angle*0.7           # This spawns the food in a weird way that points forward
                angle *=np.pi
            else:
                distance = random.random()*(FOOD_SPAWN_MAX_DISTANCE-FOOD_SPAWN_MIN_DISTANCE)+FOOD_SPAWN_MIN_DISTANCE
                angle = (random.random()-0.5)*2.*np.pi
            p = np.array([ -np.cos(angle)*distance, +np.sin(angle)*distance, random.random()+1.3])
            return p
        if avoid_locations is None:
            p = get_random_position()
        else:
            all_locations_distant = False
            while all_locations_distant is False:
                p = get_random_position()
                all_locations_distant = True
                for location in avoid_locations:
                    dist = np.linalg.norm(p[:2]-location[:2])
                    if dist < FOOD_DISTANCE_THRESHOLD:
                        all_locations_distant = False
                        break

        body_name = 'food_body_'+str(food_idx+1)
        geom_name = 'food_geom_'+str(food_idx+1)
        joint_name = 'food_free_'+str(food_idx+1)
        for axis in range(3):
            self.data.joint(joint_name).qpos[axis] = p[axis]
            self.data.body(body_name).xpos[axis] = p[axis]
            self.data.geom(geom_name).xpos[axis] = p[axis]
            self.data.body(body_name).xipos[axis] = p[axis]
            self.model.geom(geom_name).pos[axis] = p[axis]
        return p

    def respawn_robot(self):
        distance_p = random.random()*ROBOT_SPAWN_MAX_DISTANCE
        angle_p = random.random()*2.*np.pi
        p = [ -np.cos(angle_p)*distance_p, +np.sin(angle_p)*distance_p, 0.3]
        orientation = random.random()*2.*np.pi
        self.data.joint("root").qpos[:] = p + [np.sin(orientation/2), 0, 0, np.cos(orientation/2)]
        return p


    def handle_food(self):
        for contact in self.data.contact:
            if contact.geom1 in self.ignore_contact_ids:
                continue
            elif contact.geom2 in self.ignore_contact_ids:
                continue
            else:
                g1 = self.data.geom(contact.geom1)
                if g1.name in self.ignore_contact_names:
                    self.ignore_contact_ids.append(contact.geom1)
                    continue
                else:
                    g2 = self.data.geom(contact.geom2)
                    if g2.name in self.ignore_contact_names:
                        self.ignore_contact_ids.append(contact.geom2)
                        continue

                beak, food = sorted([g1.name, g2.name])
                if beak != "beak":
                    continue
                if food.startswith("food"):
                    food = int(food.split('_')[-1])
                    self.respawn_food(food-1)
                    return True
        return False

    def step(self, action):
        # Uncommenting this makes the simulation real-time-ish
        # while time.time()-self._timestamp<1./RENDER_FPS:
            # time.sleep(0.002)
        self._timestamp = time.time()
        self.do_simulation(action*30,  self.frame_skip)


        got_food = self.handle_food()
        if got_food:
            self._health += INIT_HEALTH
            reward = 1
        else:
            reward = 0

        self._steps += 1
        self._health -= 1
        if self._steps >= MAX_STEPS or self._health < 0:
            self._done = True
        self.terminated = self._done

        # Reward calculation
        current_xpos = np.array(self.data.body("torso").xpos)
        if self.previous_xpos is None:
            self.previous_xpos = np.array(current_xpos)
        xpos_inc = current_xpos - self.previous_xpos
        reward = xpos_inc
        energy = np.sum(np.abs(action/1000))
        # FANCY_REWARD = False
        # if FANCY_REWARD:
        #     if xpos_inc[0] > 0:
        #         reward = xpos_inc[0]/(energy+1)
        #     else:
        #         reward = xpos_inc[0]*(energy+1)
        # else:
        #     reward = xpos_inc[0]
        self.previous_xpos = np.array(current_xpos)


        # observation
        if self.render_cameras:
            for camera in ["view"]:
            # for camera in ["left_view", "right_view"]:
                self.camera_name = camera
                self.render_mode = "rgb_array"
                self.images[self.camera_name] = self.render()

        observation = self._get_obs()
        self.info = {
            "done": self._done,
            "reward": reward,
            "energy": energy,
            "delta": xpos_inc[0]
        }

        # render
        if self.render_view:
            self.render_mode = "human"
            self.camera_name = None
            self.render()

        self.reward = reward
        return observation, reward, self._done, self.terminated, self.info

    def _get_obs(self):
        def normalise(array, maximum):
            r = array/maximum
            r[r>maximum] = maximum
            r[r<-maximum] = -maximum
            return r

        with torch.no_grad():
            imgs =  self.images["view"].astype(np.float32)/255
            imgs = imgs.transpose(2,0,1)
            imgs = np.expand_dims(imgs, axis=0)
            compress = torch.tensor(imgs, dtype=torch.float32)
            z = vae.get_z(compress.to(device)).cpu()

        gyro = 3
        accelerometer = 3
        magnetometer = 3
        joint_pos_offset = 0
        joint_vel_offset = joint_pos_offset + 0#NUMBER_OF_JOINTS
        gyro_offset = joint_vel_offset + NUMBER_OF_JOINTS
        accelerometer_offset = gyro_offset + gyro
        magnetometer_offset = accelerometer_offset + accelerometer
        self.last_obs = {
            # "joint_position": normalise(np.array(self.data.sensordata[joint_pos_offset:joint_vel_offset]), np.pi/2),
            "joint_velocity": normalise(np.array(self.data.sensordata[joint_vel_offset:gyro_offset]), 20),
            "gyro":           normalise(np.array(self.data.sensordata[gyro_offset:accelerometer_offset]), 8),
            "accelerometer":  normalise(np.array(self.data.sensordata[accelerometer_offset:magnetometer_offset]), 100),
            "magnetometer":   normalise(np.array(self.data.sensordata[magnetometer_offset:]), 0.5),
            "z":              normalise(np.array(z), 5),
            }
        self.last_obs["joint_velocity"] = np.concatenate((self.last_obs["joint_velocity"], self.last_obs["joint_velocity"]))
        self.last_obs["gyro"]           = np.concatenate((self.last_obs["gyro"],           self.last_obs["gyro"]))
        self.last_obs["accelerometer"]  = np.concatenate((self.last_obs["accelerometer"],  self.last_obs["accelerometer"]))
        self.last_obs["magnetometer"]   = np.concatenate((self.last_obs["magnetometer"],   self.last_obs["magnetometer"]))

        # if self.render_cameras is True:
            # self.last_obs["left_view"] = self.images["left_view"],
            # self.last_obs["right_view"] = self.images["right_view"],
            # self.last_obs["view"] = self.images["view"],

        return self.last_obs

    def respawn_all_food(self):
        avoid_locations = []
        for i in range(FOOD_ITEMS):
            avoid_locations.append(self.respawn_food(i, avoid_locations, FOOD_DISTANCE_THRESHOLD))

    def reset_model(self):
        self.previous_xpos = None
        self._steps = 0
        self._health = INIT_HEALTH
        self._done = False

        self.respawn_robot()
        self.respawn_all_food()
        self._steps = 0
        self._done = False
        self._timestamp = time.time()
        self.previous_xpos = None
        observation = self._get_obs()
        return observation

    def render(self):
        return self.mujoco_renderer.render(self.render_mode, self.camera_id, self.camera_name)

    def close(self):
        self.mujoco_renderer.close()




