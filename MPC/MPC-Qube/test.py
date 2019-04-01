# coding: utf-8
import gym
import torch.utils.data as data
from dynamics import *
from controller import *
from utils import *
from quanser_robots.common import GentlyTerminating
import time

def test(mpc, model):
    reward_episodes = []
    for i in range(data_fac.n_mpc_episodes):
        data_tmp = []
        label_tmp = []
        reward_episode = 0
        state_old = data_fac.env.reset()
        for j in range(data_fac.n_max_steps):
            env.render()
            action = mpc.act(state_old, model)
            action = np.array([action])
            data_tmp.append(np.concatenate((state_old, action)))
            state_new, reward, done, info = data_fac.env.step(action)
            reward_episode += reward
            label_tmp.append(state_new - state_old)
            if done:
                break
            state_old = state_new
        reward_episodes.append(reward_episode)
        print(f"Episode [{i}/{data_fac.n_mpc_episodes}], Reward: {reward_episode:.8f}")
    return reward_episodes

env_id ="Qube-v0" # "CartPole-v0"
env = GentlyTerminating(gym.make(env_id))
config_path = "config.yml"
config = load_config(config_path)
print_config(config_path)

config["model_config"]["load_model"] = True
config["dataset_config"]["load_flag"] = True

model = DynamicModel(config)

data_fac = DatasetFactory(env,config)
model.norm_train_data(data_fac.all_dataset["data"],data_fac.all_dataset["label"])

mpc = MPC(env,config)

rewards_list = []
for itr in range(config["dataset_config"]["n_mpc_itrs"]):
    t = time.time()
    print("**********************************************")
    print("The reinforce process [%s], collecting data ..." % itr)
    rewards = test(mpc, model)
    rewards_list += rewards
    plt.close("all")
    plt.figure(figsize=(12, 5))
    plt.title('Reward Trend with %s iteration' % itr)
    plt.plot(rewards_list)
    plt.savefig("storage/reward-" + str(model.exp_number) + "_test.png")
    print("Consume %s s in this iteration" % (time.time() - t))
    loss = model.trai
