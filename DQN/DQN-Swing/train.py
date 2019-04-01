# coding: utf-8

from DQN import *
import gym
from quanser_robots.common import GentlyTerminating


def train():
    '''Load the configuration setttings'''
    config_path = "config.yml"
    print_config(config_path)
    config = load_config(config_path)
    training_config = config["training_config"]
    seed = training_config["random_seed"]
    n_episodes = training_config["n_episodes"]
    max_episode_step = training_config["max_episode_step"]
    n_update_target = training_config["n_update_target"]
    exp_number = training_config["exp_number"]
    save_model_path = training_config["save_model_path"]
    render_flag = training_config["render"]
    save_best = training_config["save_best"]
    save_thres = training_config["save_thres"]

    '''Use fixed epsilon or use a exponential function decay?'''
    if training_config["use_fix_epsilon"]:
        epsilon_by_frame = lambda frame_idx: training_config["fix_epsilon"]
    else:
        epsilon_start = training_config["epsilon_start"]
        epsilon_final = training_config["epsilon_final"]
        epsilon_decay = training_config["epsilon_decay"]
        epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * np.exp(
            -1. * frame_idx / epsilon_decay)
    torch.manual_seed(seed)
    np.random.seed(seed)

    '''Environment initialization'''
    env_id = "CartpoleSwingShort-v0"
    env = GentlyTerminating(gym.make(env_id))

    '''Initialize the DQN algorithm object'''
    policy = Policy(env,config)
    losses = []
    all_rewards = []
    avg_rewards = []
    epsilons = []

    '''Training the q-network with n episodes'''
    for i_episode in range(n_episodes):
        episode_reward = 0
        state = env.reset()
        state[4] /= 10
        epsilon = epsilon_by_frame(i_episode)
        epsilons.append(epsilon)
        for step in range(max_episode_step):
            if render_flag:
                env.render()
            '''Choose action'''
            action = policy.act(state, epsilon)
            f_action = 5*(action-(policy.n_actions-1)/2)/((policy.n_actions-1)/2)
            next_state, reward, done, _ = env.step(f_action)
            next_state[4]/=10
            policy.replay_buffer.push(state, action[0], reward, next_state, done)
            state = next_state
            episode_reward += reward
            if done:
                break
            if len(policy.replay_buffer) > policy.batch_size:
                loss = policy.train()
                losses.append(loss.item())

        all_rewards.append(episode_reward)
        avg_rewards.append(np.mean(all_rewards[-5:]))

        if i_episode % 4 == 0:
            save_fig(i_episode, all_rewards,avg_rewards, losses,epsilons, exp_number)
            print("Exp %s, episode %s, avg episode reward %s" % (exp_number, i_episode, np.mean(all_rewards[-10:])))

        if i_episode % n_update_target == 0:
            '''Update the target network'''
            policy.update_target()

        policy.save_model(save_model_path)
        if save_best and i_episode>10 and episode_reward>save_thres:
            ratio = 1.1
            if episode_reward > ratio*np.mean(all_rewards[-10:]):
                print("Save model with episode reward %s " % (episode_reward))
                print("Model path: %s " % (save_model_path))
                break
    env.close()

if __name__ =="__main__":
    train()