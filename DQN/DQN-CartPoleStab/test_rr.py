# coding: utf-8

from DQN import *
import argparse


use_plot = True
render = True

window = 500
collect_fr = 10
plot_fr = 10
render_fr = 10

if use_plot:
    plt.ion()
    plot = PlotSignal(window=window)

# Initialize Controller & Environment:
env, ctrl = get_env_and_controller(long_pendulum=False, simulation=True, swinging=False, mouse_control=False)


config_path = "config.yml"
print_config(config_path)
config = load_config(config_path)
training_config = config["training_config"]
config["model_config"]["load_model"] = True

n_episodes = 10
max_episode_step = 100000
print("*********************************************")
print("Testing the model on real platform for 10 episodes with 100000 maximum steps per episode")
print("*********************************************")

policy = Policy(env,config)
losses = []
all_rewards = []
avg_rewards = []
epsilons = []


for i in range(n_episodes):
    print("\n\n###############################")
    print("Episode {0}".format(0))

    # Reset the environment:
    env.reset()
    obs, reward, done, _ = env.step(np.zeros(1))
    # Start the Control Loop:
    print("\nStart Controller:\t\t\t", end="")
    for n in range(max_episode_step):
        action = policy.act(obs, 0)
        f_action = 12 * (action - (policy.n_actions - 1) / 2) / ((policy.n_actions - 1) / 2)
        obs, reward, done, _ = env.step(f_action)
        all_rewards.append(reward)
        if done:
            print("Physical Limits or End of Time reached")
            break

        if render and np.mod(n, render_fr) == 0:
            env.render()

        if use_plot and np.mod(n, collect_fr) == 0:
            alpha, theta = get_angles(obs[1], obs[2])
            plot.update(theta=theta, alpha=alpha, theta_dt=obs[4], volt=f_action, u=0, x=obs[0])
            env.render()

        if use_plot and np.mod(n, plot_fr) == 0:
            plot.plot_signal()

    # Stop the cart:
    env.step(np.zeros(1))

print("avg reward: ",np.mean(all_rewards))
print("rewards: ", all_rewards)
env.close()



