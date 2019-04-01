import yaml
import os
import matplotlib.pyplot as plt
import numpy as np

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