import sys
import time
import pickle
import numpy as np

import survival
import gymnasium as gym


import pygame

pygame.init()
pygame.joystick.init()
joystick_count = pygame.joystick.get_count()
joystick = pygame.joystick.Joystick(1)
joystick.init()
axes = joystick.get_numaxes()

try:
    with open('joystick_calibration.pickle', 'rb') as f:
        centre, values, min_values, max_values = pickle.load(f)
except:
    centre = {}
    values = {}
    min_values = {}
    max_values = {}
    for axis in range(joystick.get_numaxes()):
        values[axis] = 0.
        centre[axis] = 0.
        min_values[axis] = 0.
        max_values[axis] = 0.
    T = 3.
    print(f'Leave the controller neutral for {T} seconds')
    t = time.time()
    while time.time() - t < T:
        pygame.event.pump()
        for axis in range(axes):
            centre[axis] = joystick.get_axis(axis)
        time.sleep(0.05)
    T = 5.
    print(f'Move the joystick around for {T} seconds trying to reach the max and min values for the axes')
    t = time.time()
    while time.time() - t < T:
        pygame.event.pump()
        for axis in range(axes):
            value = joystick.get_axis(axis)-centre[axis]
            if value > max_values[axis]:
                max_values[axis] = value
            if value < min_values[axis]:
                min_values[axis] = value
        time.sleep(0.05)
    with open('joystick_calibration.pickle', 'wb') as f:
        pickle.dump([centre, values, min_values, max_values], f)
print(min_values)
print(max_values)



env = survival.SurvivalEnv(render_view=True, render_cameras=False)

values = [0]*axes
while True:
    env.reset()
    
    
    while True:
        pygame.event.pump()
        for i in range(joystick_count):
            for axis in range(axes):
                values[axis] = joystick.get_axis(axis)-centre[axis]

        action = -np.array([values[1]-values[2], values[1]+values[2]])
        obs, reward, restart, _, info = env.step(action)
        env.render()

