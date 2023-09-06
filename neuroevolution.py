import os
import sys

import numpy as np

import survival
import gymnasium as gym

import time
import glob
import random
import datetime

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

import cma

ENV_NAME = 'survivalenv'
EPISODES_PER_GENERATION = 25
GENERATIONS = 10000
POPULATION_SIZE = 80
SIGMA = 1.
SAVE_PATH = "./"


class NeuralNetwork(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(NeuralNetwork, self).__init__()
        self.l1 = nn.Linear(input_shape, 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, 64)
        self.lout = nn.Linear(64, n_actions)

    def forward(self, x):
        x = F.elu(self.l1(x.float()))
        x = F.elu(self.l2(x))
        x = F.elu(self.l3(x))
        return torch.tanh(self.lout(x))

    def get_params(self):
        p = np.empty((0,))
        for n in self.parameters():
            p = np.append(p, n.flatten().cpu().detach().numpy())
        return p

    def set_params(self, x):
        start = 0
        for p in self.parameters():
            e = start + np.prod(p.shape)
            p.data = torch.FloatTensor(x[start:e]).reshape(p.shape)
            start = e


def evaluate(ann, env, seed, render=False, wait_after_render=False):
    obs = env.reset()
    obs = preprocess_observation(obs[0])
    prev_obs = obs
    total_reward = 0
    while True:
        if render is True:
            env.render()
        # Output of the neural net
        cat = torch.cat((prev_obs, obs), 0)
        net_output = ann(cat)
        # the action is the value clipped returned by the nn
        action = net_output.data.cpu().numpy()

        prev_obs = obs.clone()
        obs, reward, done, _, _ = env.step(action)
        obs = preprocess_observation(obs)
        total_reward += reward
        if done:
            break
    if render and wait_after_render:
        for i in range(2):
            env.render()
            time.sleep(1)
    return total_reward


def fitness(candidate, env, seed, render=False):
    ann.set_params(candidate)
    return -evaluate(ann, env, seed, render)


def preprocess_observation(obs):
    return torch.cat([torch.tensor(obs[o]).view(-1) for o in sorted(obs.keys())], 0)


def train_with_cma(generations, writer_name):
    es = cma.CMAEvolutionStrategy(len(ann.get_params()) * [0], SIGMA, {'popsize': POPULATION_SIZE})
    # es = cma.CMAEvolutionStrategy(len(ann.get_params())*[0], SIGMA, {'popsize': POPULATION_SIZE, 'seed': 123})
    best = 0

    def reps(generation):
        if generation < 20:
            return 1
        elif generation < 50:
            return 2
        elif generation < 100:
            return 4
        elif generation < 200:
            return 5
        elif generation < 300:
            return 6
        elif generation < 400:
            return 7
        elif generation < 500:
            return 8
        else:
            return EPISODES_PER_GENERATION

    def scale(generation):
        return 1
        # if generation < 10:
        #     return 1
        # elif generation < 50:
        #     return 0.9
        # elif generation < 100:
        #     return 0.8
        # elif generation < 200:
        #     return 0.7
        # elif generation < 300:
        #     return 0.6
        # elif generation < 400:
        #     return 0.4
        # elif generation < 500:
        #     return 0.2
        # else:
        #      return 0.1

    for generation in range(generations):
        seeds = [random.getrandbits(32) for _ in range(reps(generation))]
        # Create population of candidates and evaluate them
        candidates, fitnesses, Maxfitnesses = es.ask(sigma_fac=scale(generation)), [], []
        for candidate in candidates:
            reward = 0
            for seed in seeds:
                # Evaluate the agent using stable-baselines predict function
                reward += fitness(candidate, env, seed, render=False)
            average_candidate_reward = reward / reps(generation)
            fitnesses.append(average_candidate_reward)
            Maxfitnesses.append(-average_candidate_reward)
        # CMA-ES update
        es.tell(candidates, fitnesses)

        # Display some training infos
        mean_fitness = np.mean(sorted(fitnesses)[:int(0.1 * len(candidates))])
        print("Iteration {:<3} Mean top 10% reward: {:.2f}".format(generation, -mean_fitness))
        cur_best = max(Maxfitnesses)
        best_index = np.argmax(Maxfitnesses)
        # print("current  value {}...".format(cur_best))
        writer.add_scalar('performance/mean top 10 reward', -mean_fitness, generation)
        writer.add_scalar('params/reps', reps(generation), generation)
        writer.add_scalar('params/sigma', SIGMA * scale(generation), generation)

        best_params = candidates[best_index]
        render_the_test = os.path.exists("render")
        seeds = [random.getrandbits(32) for _ in range(EPISODES_PER_GENERATION)]
        test_rew = 0
        ann.set_params(best_params)
        for seed in seeds:
            test_rew += evaluate(ann, env, seed, render=render_the_test)
        test_rew /= EPISODES_PER_GENERATION
        writer.add_scalar('performance/test reward', test_rew, generation)
        # Save model if it's best
        if not best or test_rew >= best:
            best = test_rew
            print("Saving new best with value {}...".format(best))
            torch.save(ann.state_dict(), writer_name + '_BEST.pth')
        # Saving model every
        if (generation + 1) % 5 == 0:
            try:
                torch.save(ann.state_dict(),
                           os.path.join(SAVE_PATH, writer_name + "_gen_" + str(generation + 1).zfill(8) + ".pth"))
            except:
                print("Error in saving model")

    print('best reward : {}'.format(best))


if __name__ == '__main__':
    # if len(sys.argv)>2 and sys.argv[1] == '-test':
    env = survival.SurvivalEnv(render_cameras=True, render_view=True)
    # else:
    # env = survival.SurvivalEnv(render_cameras=False, render_view=False)
    raw_obs = env.reset()[0]
    obs = preprocess_observation(raw_obs)
    n_joints = survival.NUMBER_OF_JOINTS

    print(f'Joints {n_joints}')
    ann = NeuralNetwork(obs.shape[0] * 2, n_joints)
    print(f'NN {obs.shape[0] * 2} {n_joints}')

    if len(sys.argv) > 2 and sys.argv[1] == '-test':
        env.render_mode = "human"
        ann.load_state_dict(torch.load(sys.argv[2]))
        accumulated_reward = 0
        done = 0
        while True:
            reward = evaluate(ann, env, seed=random.getrandbits(32), render=True, wait_after_render=True)
            accumulated_reward += reward
            done += 1
            print(f'Reward: {reward}    (avg:{accumulated_reward / done})')
    else:
        # while not os.path.exists("start"):
        # time.sleep(1)

        if not os.path.isdir(SAVE_PATH):
            os.makedirs(SAVE_PATH)

        np.random.seed(123)
        now = datetime.datetime.now()
        date_time = "{}_{}.{}.{}".format(now.day, now.hour, now.minute, now.second)
        writer_name = f'cmaC3_{ENV_NAME}_pop{POPULATION_SIZE}_k{EPISODES_PER_GENERATION}_sigma{SIGMA}_{date_time}'
        writer = SummaryWriter(log_dir='runs/' + writer_name)

        train_with_cma(GENERATIONS, writer_name)

