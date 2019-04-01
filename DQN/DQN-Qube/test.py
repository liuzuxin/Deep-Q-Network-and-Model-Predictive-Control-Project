# coding: utf-8

from DQN import *
import gym
from quanser_robots.common import GentlyTerminating

def test():
    config_path = "config.yml"
    print_config(config_path)
    config = load_config(config_path)
    training_config = config["training_config"]
    config["model_config"]["load_model"] = True

    env_id ="Qube-v0"
    env = GentlyTerminating(gym.make(env_id))

    n_episodes = 10
    max_episode_step = 10000
    print("*********************************************")
    print("Testing the model for 10 episodes with 10000 maximum steps per episode")
    print("*********************************************")

    policy = Policy(env,config)

    losses = []
    all_rewards = []
    avg_rewards = []
    epsilons = []
    for i_episode in range(n_episodes):
        episode_reward = 0
        state = env.reset()
        state[4:6]/=20
        epsilon = 0
        epsilons.append(epsilon)
        for step in range(max_episode_step):
            env.render()
            action = policy.act(state, epsilon)
            f_action = 5*(action-(policy.n_actions-1)/2)/((policy.n_actions-1)/2)
            next_state, reward, done, _ = env.step(f_action)
            reward = 100*reward
            next_state[4:6]/=20
            policy.replay_buffer.push(state, action[0], reward, next_state, done)
            state = next_state
            episode_reward += reward
            if done:
                break
        all_rewards.append(episode_reward)
        avg_rewards.append(np.mean(all_rewards[-3:]))
    plot_fig(n_episodes, all_rewards,avg_rewards, losses)
    env.close()

if __name__ =="__main__":
    test()
