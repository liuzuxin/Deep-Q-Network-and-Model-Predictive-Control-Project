# DQN - Deep Q-Network

This folder contains the implementation of DQN algorithm and the evaluation of it.

For more details of DQN, see the paper [here](https://arxiv.org/abs/1312.5602)

Choose the environment folder and follow the instructions to run everything.

## Overview of the experiment results:

The best experients parameters in different environments:

| Environment  | Learning Rate    | Epsilon Decay  |  Batch Size  |  Action Number   | Gamma  |   average episode reward |
| --------   | -----:  | :----: | :----: | :----: | :----: |  :----: |
| Qube      |  0.001      |   1000    | 50   |   9  | 0.99  | 410  |
| CartPole Swingup |  0.001      |   1000    |  64  |  7   | 0.995  |  4126 |
| CartPole Stab   | 0.001      |   500   |   64 |  9  | 0.995  |  1535  |
| Double CartPole    | 0.001     |  2000   |   64 |  7   | 0.995  |  383  |




### CartpoleStabShort-v0
   episode_rewards:
   
   learning_rate: 3e-5
   
   networks architecture:
   
   gamma: 0.98
   
   batch size: 20
   
   weight_decay: 1e-4
   
   num_epochs: 2000
   
### Qube-v0:
   episode_rewards:
   
   learning_rate: 3e-5
   
   networks architecture:
   
   gamma: 0.98
   
   batch size: 20
   
   weight_decay: 1e-4
   
   num_epochs: 2000
   
   
### DoublePendulum-v0
   episode_rewards:
   
   learning_rate: 3e-5
   
   networks architecture:
   
   gamma: 0.99
   
   batch size: 20

   weight_decay: 1e-4
   
   num_epochs: 2000

### CartpoleSwingShort-v0
   learning_rate: 3e-5
   
   networks architecture:
   
   gamma: 0.98
   
   batch size: 20
   
   weight_decay: 1e-4
   
   num_epochs: 2000

