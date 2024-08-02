"""

    postprocessing for the first example of a NN with a single layer in equinox

    I should also re-create this with JAX alone  
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from function_approximation.model import (
    Linear, 
    NonLinear, 
    P_1_FE,
    SingleLayer
)
from function_approximation.helpers import parse_nn_yaml
import function_approximation.model as models

import matplotlib.pyplot as plt

if __name__ == "__main__":

    # should make this not hardcoded asap
    data_dict = parse_nn_yaml("examples/first_examples/single_layer_approx/options.yaml")
    nn_params = data_dict["Network"]["options"]
    results_dir = data_dict["Project"]["save_dir"]

    # is used to generate the model, so is inherently flexible 
    model_type = data_dict["Network"]["name"]

    # might need to access these in this main script
    batch_size = nn_params["batch_size"]
    in_size = nn_params["in_size"]
    out_size = nn_params["out_size"]
    layer_width = nn_params["layer_width"]
    
    loss = jnp.load(results_dir + model_type + "_loss.npy")

    # not sure if this is the optimal way to convert jax array to numpy array
    loss = np.asarray(loss)


    # already know what x and y are, might want to save functions

    x_s = np.arange(0, len(loss))

    # plot the loss
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.semilogy(x_s, loss)
    plt.show()

    ###### use the nn to compute f_{NN}(x), compare to f(x) #######
    # load the model
    model_name = results_dir + model_type + "_model.eqx"
    
    # hardcoding in_size and out_size but these can and should be read in from
    # the yaml file
    in_size, out_size = 1, 1

    # define model using getattr()
    try:
        # this might be slightly redundant
        model_class = getattr(models, model_type)
        model = model_class(jax.random.PRNGKey(0), **nn_params)
    except:
        raise Exception("The model you requested is not implemented,\
                            check function_approxmiation.models for existing NN classes")

    model = eqx.tree_deserialise_leaves(model_name, model)

    # plot the prediction vs. the actual function
    x_i = np.tile(jnp.linspace(0,1,batch_size), (1,1)).T
    x_i_true = np.tile(jnp.linspace(0,1,100), (1,1)).T # increase density for true sol
    
    y_pred = jax.vmap(model)(x_i)
    y_true = x_i_true ** 2              # again, this is specific to this problem

    # plot the result!
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(x_i, y_pred, 'bo-')
    ax.plot(x_i_true, y_true, 'r')
    ax.legend(["Predicted", "True"])
    plt.show()