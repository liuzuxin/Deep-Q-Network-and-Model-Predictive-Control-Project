# MPC - Model Predictive Control

This folder contains the implementation of MPC algorithm and the evaluation of it.

The implementation is mainly followed in this paper [here](https://ieeexplore.ieee.org/abstract/document/8463189)

To optimize the MPC controller, we use the [Artificial Bee Colony](https://en.wikipedia.org/wiki/Artificial_bee_colony_algorithm) (ABC) optimization algorithm, 
instead of the original random shooting method in the paper. The implementation of ABC algorithm is based on this repo: [https://github.com/rwuilbercq/Hive](https://github.com/rwuilbercq/Hive)

Choose the environment folder and follow the instructions to run everything.

Jupyter notebook example is in the ```./MPC-CartPoleStab``` folder.

## Overview of the experiment results:


The best results in different environments:

| Environment  | Horizon   |Numb\_bees  |   Max\_itrs  |  Gamma  |  Episode reward |
| --------   | -----:  | :----: | :----: | :----: |  :----: |
| Qube      |  30    |  8  | 30  |  0.98  | 4.0 |
| CartPole Swingup |  20  | 8   | 20 |  0.99  | 2000 |
| CartPole Stab   | 12  | 8  |  20 |  0.99  | 19999 |
| Double CartPole    | 5  | 8 |  20 |  0.99  | 91 |
