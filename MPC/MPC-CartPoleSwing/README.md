# MPC - CartPoleSwing
This folder contains the implementation of MPC algorithm and the evaluation on the CartPoleSwing environment

The implementation is mainly followed in this paper [here](https://ieeexplore.ieee.org/abstract/document/8463189)

To optimize the MPC controller, we use the [Artificial Bee Colony](https://en.wikipedia.org/wiki/Artificial_bee_colony_algorithm) (ABC) optimization algorithm, 
instead of the original random shooting method in the paper. The implementation of ABC algorithm is based on this repo: [https://github.com/rwuilbercq/Hive](https://github.com/rwuilbercq/Hive)

All the hyper-parameters and experiment setting are stored in the ```config.yml``` file

All the results (figure and model) will be stored in the ```./storage``` folder by default

If you are not familiar with this environment, you can use the  `analyze_env()`  function in the `utils.py` to help you quickly understand the environment's state space, action space, reward range, etc.

### How to run

To try our pre-trained model, simply run

```angularjs
python run.py --path config.yml
```
The script will load the configurations in the ```config.yml``` file and begin to train

Note that because of the long time of optimization, boost the data with MPC controller would take a long time

If you want to load the dataset and a pre-trained dynamic model, note that you should normalize the dataset first, because the dynamic model need the data distribution information.
You can use the `norm_train_data()` method in the `DynamicModel` class.
### Configuration explanation

In the ```config.yml``` file, there are 4 sets of configuration.

The `model_config`  part is the configuration of the parameters which determine the neural network architecture and the environment basis.

The `training_config` part is the configuration of the training process parameters.

The `dataset_config` part is the configuration of the dataset parameters.

The `mpc_config` part is the configuration of the MPC algorithm parameters.

The `exp_number` parameter in the `training_config` is the number of your experiment. The name of saved figure results in the `./storage` folder will be determined by this parameter.

If you want to train your model from scratch, then set the `load_model` parameter to `False`. If set to `True`, the trainer will load the model from `model_path`.
