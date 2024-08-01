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

import matplotlib.pyplot as plt

if __name__ == "__main__":

    # should make this not hardcoded asap
    data_dict = parse_nn_yaml("examples/first_examples/single_layer_approx/options.yaml")
    nn_params = data_dict["Network"]

    # Linear, Nonlinear, and 
    model_type = nn_params["name"]
    batch_size = nn_params["options"]["batch_size"]
    layer_width = nn_params["options"]["layer_width"]
    
    h = 2.0/float(batch_size)

    loss = jnp.load("/home/kenneth/research/learning/ML/function_approximation_data/" + model_type + "_loss.npy")

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
    model_name = "/home/kenneth/research/learning/ML/function_approximation_data/" + model_type + "_model.eqx"
    # hardcoding in_size and out_size but these can and should be read in from
    # the yaml file
    in_size, out_size = 1, 1

    # defines model - need to do this without if statements, use a function
    # or something!
    if model_type == "Linear":
        model = Linear(in_size, out_size, key=jax.random.PRNGKey(0))
    elif model_type == "Nonlinear":
        model = NonLinear(in_size, out_size, key=jax.random.PRNGKey(0))
    elif model_type == "P_1_FE":
        model = P_1_FE(in_size, out_size, jax.random.PRNGKey(0), h)
    elif model_type == "SingleLayer":
        model = SingleLayer(in_size, out_size, layer_width, jax.random.PRNGKey(0))  
    else:
        raise Exception("You're trying to use a model I haven't implemented yet")

    model = eqx.tree_deserialise_leaves(model_name, model)

    # plot the prediction vs. the actual function
    x_i = np.tile(jnp.linspace(0,1,20), (1,1)).T
    y_pred = jax.vmap(model)(x_i)
    y_true = x_i ** 2              # again, this is specific to this problem

    # plot the result!
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(x_i, y_pred, 'b')
    ax.plot(x_i, y_true, 'r')
    ax.legend(["Predicted", "True"])
    plt.show()

    # access the model's weights and biases to see if anything funky is going on
    print(model)



