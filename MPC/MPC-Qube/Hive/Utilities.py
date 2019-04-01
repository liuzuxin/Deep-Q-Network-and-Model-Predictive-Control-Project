#!/usr/bin/env python

# ---- MODULE DOCSTRING

__doc__ = """

(C) Hive, Romain Wuilbercq, 2017
     _
    /_/_      .'''.
 =O(_)))) ...'     `.
    \_\              `.    .'''X
                       `..'
.---.  .---..-./`) ,---.  ,---.   .-''-.
|   |  |_ _|\ .-.')|   /  |   | .'_ _   \
|   |  ( ' )/ `-' \|  |   |  .'/ ( ` )   '
|   '-(_{;}_)`-'`"`|  | _ |  |. (_ o _)  |
|      (_,_) .---. |  _( )_  ||  (_,_)___|
| _ _--.   | |   | \ (_ o._) /'  \   .---.
|( ' ) |   | |   |  \ (_,_) /  \  `-'    /
(_{;}_)|   | |   |   \     /    \       /
'(_,_) '---' '---'    `---`      `'-..-'

Description:
-----------

A series of utility functions (such as plotting function etc...).

"""

# ---- IMPORT MODULES

try:
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
except:
    raise ImportError("Install 'matplotlib' to plot convergence results.")

# ---- CONVERGENCE PLOT

def ConvergencePlot(cost):
    """

    Monitors convergence.

    Parameters:
    ----------

        :param dict cost: mean and best cost over cycles/generations as returned
                          by an optimiser.

    """
    plt.rc('font',family=  'Tibetan Machine Uni')
    fs=15
    font = FontProperties();
    font.set_size( fs) #'larger');
    labels = ["Best Bee's Cost", "Mean Bees' Cost"]
    plt.figure(figsize=(7, 4.5));
    plt.plot(range(len(cost["best"])), cost["best"], label=labels[0]);
    plt.scatter(range(len(cost["mean"])), cost["mean"], color='red', label=labels[1]);
    plt.xlabel("Iteration",fontsize=fs);
    plt.ylabel("Cost",fontsize=fs);
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(loc="best", prop = font);
    plt.xlim([0,len(cost["mean"])]);
    plt.grid();
    plt.savefig("mpc.png")
    plt.show();

# ---- END
