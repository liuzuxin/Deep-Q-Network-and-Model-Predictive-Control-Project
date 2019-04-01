# coding: utf-8
import gym
import argparse
from dynamics import *
from controller import *
from utils import *
from quanser_robots.common import GentlyTerminating
import time

parser = argparse.ArgumentParser(description='Specify the configuraton file path')
parser.add_argument('--path', required=False, type=str, default='config.yml',
                    help='Specify the configuraton file path')


args = parser.parse_args()

config_path = args.path # "config.yml"
config = load_config(config_path)
print_config(config_path)

env_id = "CartpoleSwingShort-v0"
env = GentlyTerminating(gym.make(env_id))

model = DynamicModel(config)

data_fac = DatasetFactory(env,config)
data_fac.collect_random_dataset()

loss = model.train(data_fac.random_trainset,data_fac.random_testset)
model.plot_model_validation(env,n_sample=200)
mpc = MPC(env,config)

rewards_list = []
for itr in range(config["dataset_config"]["n_mpc_itrs"]):
    t = time.time()
    print("**********************************************")
    print("The reinforce process [%s], collecting data ..." % itr)
    rewards = data_fac.collect_mpc_dataset(mpc, model)
    trainset, testset = data_fac.make_dataset()
    rewards_list += rewards

    plt.close("all")
    plt.figure(figsize=(12, 5))
    plt.title('Reward Trend with %s iteration' % itr)
    plt.plot(rewards_list)
    plt.savefig("storage/reward-" + str(model.exp_number) + ".png")
    print("Consume %s s in this iteration" % (time.time() - t))
    loss = model.train(trainset, testset)
