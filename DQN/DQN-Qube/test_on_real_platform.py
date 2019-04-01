# coding: utf-8

from DQN import *
import argparse
from quanser_robots import GentlyTerminating

plt.style.use('seaborn')
env = GentlyTerminating(gym.make('QubeRR-v0'))

config_path = "config.yml"
print_config(config_path)
config = load_config(config_path)
training_config = config["training_config"]
config["model_config"]["load_model"] = True

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

s_all = []
a_all = []

for i in range(n_episodes):
    print("Testing episodes %s" %i)
    obs_old = env.reset()
    obs_old[4:6] /= 20
    done = False
    while not done:
        env.render()
        action = policy.act(obs_old, 0.0)
        f_action = 5 * (action - (policy.n_actions - 1) / 2) / ((policy.n_actions - 1) / 2)
        obs_new, reward, done, info = env.step(f_action)
        reward = 100*reward
        all_rewards.append(reward)
        obs_new[4:6] /= 20
        obs_old = obs_new
        s_all.append(info['s'])
        a_all.append(info['a'])

print("avg reward: ",np.mean(all_rewards))
print("rewards: ", all_rewards)
env.close()

fig, axes = plt.subplots(5, 1, figsize=(5, 8), tight_layout=True)

s_all = np.stack(s_all)
a_all = np.stack(a_all)

n_points = s_all.shape[0]
t = np.linspace(0, n_points * env.unwrapped.timing.dt_ctrl, n_points)
for i in range(4):
    state_labels = env.unwrapped.state_space.labels[i]
    axes[i].plot(t, s_all.T[i], label=state_labels, c='C{}'.format(i))
    axes[i].legend(loc='lower right')
action_labels = env.unwrapped.action_space.labels[0]
axes[4].plot(t, a_all.T[0], label=action_labels, c='C{}'.format(4))
axes[4].legend(loc='lower right')

axes[0].set_ylabel('ang pos [rad]')
axes[1].set_ylabel('ang pos [rad]')
axes[2].set_ylabel('ang vel [rad/s]')
axes[3].set_ylabel('ang vel [rad/s]')
axes[4].set_ylabel('voltage [V]')
axes[4].set_xlabel('time [seconds]')
plt.show()


