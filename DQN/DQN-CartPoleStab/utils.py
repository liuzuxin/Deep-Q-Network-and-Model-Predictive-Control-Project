import yaml
import os
import numpy as np
import gym
import matplotlib.pyplot as plt

from quanser_robots.cartpole.ctrl import SwingUpCtrl, MouseCtrl
from quanser_robots.common import GentlyTerminating, Logger

def load_config(config_path="config.yml"):
    '''Load the configuration setting from a given path'''

    if os.path.isfile(config_path):
        f = open(config_path)
        return yaml.load(f)
    else:
        raise Exception("Configuration file is not found in the path: "+config_path)

def print_config(config_path="config.yml"):
    '''Print the configuration setting from a given path'''

    if os.path.isfile(config_path):
        f = open(config_path)
        config = yaml.load(f)
        print("************************")
        print("*** model configuration ***")
        print(yaml.dump(config["model_config"], default_flow_style=False, default_style=''))
        print("*** train configuration ***")
        print(yaml.dump(config["training_config"], default_flow_style=False, default_style=''))
        print("************************")
    else:
        raise Exception("Configuration file is not found in the path: "+config_path)

def anylize_env(env, test_episodes = 100,max_episode_step = 500, render = False):
    '''Analyze the environment through random sampled episodes data'''

    print("state space shape: ", env.observation_space.shape)
    print("state space lower bound: ", env.observation_space.low)
    print("state space upper bound: ", env.observation_space.high)
    print("action space shape: ", env.action_space.shape)
    print("action space lower bound: ", env.action_space.low)
    print("action space upper bound: ", env.action_space.high)
    print("reward range: ", env.reward_range)
    rewards = []
    steps = []
    for episode in range(test_episodes):
        env.reset()
        step = 0
        episode_reward = 0
        for _ in range(max_episode_step):
            if render:
                env.render()
            step += 1
            action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            if done:
               # print("done with step: %s " % (step))
                break
        steps.append(step)
        rewards.append(episode_reward)
    env.close()
    print("Randomly sample actions for %s episodes, with maximum %s steps per episodes"
          % (test_episodes, max_episode_step))
    print(" average reward per episode: %s, std: %s " % (np.mean(rewards), np.std(rewards) ))
    print(" average steps per episode: ", np.mean(steps))
    print(" average reward per step: ", np.sum(rewards)/np.sum(steps))


def plot_fig(episode, all_rewards,avg_rewards, losses):
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.title('Reward Trend with %s Episodes' % (episode))
    plt.xlabel("episode")
    plt.ylabel("reward")
    plt.plot(all_rewards, 'b')
    plt.plot(avg_rewards, 'r')
    plt.subplot(122)
    plt.title('Loss Trend with %s Episodes' % (episode))
    plt.plot(losses)
    plt.show()

def plot(frame_idx, rewards, losses):
    plt.clf()
    plt.close()
    plt.ion()
    plt.figure(figsize=(12 ,5))
    plt.subplot(131)
    plt.title('episode %s. reward: %s' % (frame_idx, np.mean(rewards[-10:])))
    plt.plot(rewards)
    plt.subplot(132)
    plt.title('loss')
    plt.plot(losses)
    plt.pause(0.0001)

def save_fig(episode, all_rewards, avg_rewards, losses, epsilon, number = 0):
    '''Save the experiment results in the ./storage folder'''
    plt.clf()
    plt.close("all")
    plt.figure(figsize=(8 ,5))
    plt.title('Reward Trend with %s Episodes' % (episode))
    plt.xlabel("episode")
    plt.ylabel("reward")
    plt.plot(all_rewards,'b')
    plt.plot(avg_rewards,'r')
    plt.savefig("storage/reward-"+str(number)+".png")
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.title('Loss Trend with Latest %s Steps' % (1200))
    plt.plot(losses[-1200:])
    plt.subplot(122)
    plt.title('Epsilon with %s Episodes' % (episode))
    plt.plot(epsilon)
    plt.savefig("storage/loss-"+str(number)+".png")


def get_angles(sin_theta, cos_theta):
    theta = np.arctan2(sin_theta, cos_theta)
    if theta > 0:
        alpha = (-np.pi + theta)
    else:
        alpha = (np.pi + theta)
    return alpha, theta


class PlotSignal:
    def __init__(self, window=10000):
        self.window = window
        self.values = {}

    def update(self, **argv):
        for k in argv:
            if k not in self.values:
                self.values[k] = [argv[k]]
            else:
                self.values[k].append(argv[k])
            self.values[k] = self.values[k][-self.window:]

    def plot_signal(self):
        N = len(self.values)
        plt.clf()
        for i, k in enumerate(self.values):
            plt.subplot(N, 1, i + 1)
            plt.title(k)

            plt.plot(self.values[k])

        plt.pause(0.0000001)

    def last_plot(self):
        N = len(self.values)
        plt.clf()
        plt.ioff()
        for i, k in enumerate(self.values):
            plt.subplot(N, 1, i + 1)
            plt.title(k)

            plt.plot(self.values[k])

        plt.show()


def do_trajectory(env, ctrl, plot, time_steps=10000, use_plot=True,
                  collect_fr=10, plot_fr=10, render=True, render_fr=10):

    obs = env.reset()
    for n in range(time_steps):
        act = ctrl(obs)
        obs, _, done, _ = env.step(np.array(act[0]))

        if done:
            print("Physical Limits or End of Time reached")
            break

        if render:
            if n % render_fr == 0:
                env.render()

        if use_plot:

            if n % collect_fr == 0:
                alpha, theta = get_angles(obs[1], obs[2])
                plot.update(theta=theta, alpha=alpha, theta_dt=obs[4], volt=act[0], u=act[1], x=obs[0])
                env.render()

            if n % plot_fr == 0:
                plot.plot_signal()


def get_env_and_controller(long_pendulum=True, simulation=True, swinging=True, mouse_control=False):
    pendulum_str = {True:"Long", False:"Short"}
    simulation_str = {True:"", False:"RR"}
    task_str = {True:"Swing", False:"Stab"}

    if not simulation:
        pendulum_str = {True: "", False: ""}

    mu = 7.5 if long_pendulum else 19.
    env_name = "Cartpole%s%s%s-v0" % (task_str[swinging], pendulum_str[long_pendulum], simulation_str[simulation])
    if not mouse_control:
        return Logger(GentlyTerminating(gym.make(env_name))), SwingUpCtrl(long=long_pendulum, mu=mu)
    else:
        return Logger(GentlyTerminating(gym.make(env_name))), MouseCtrl()

