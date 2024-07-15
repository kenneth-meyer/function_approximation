import jax
import optax

import equinox as eqx

# functions in this repo
from function_approximation.model import Linear
from function_approximation.helpers import parse_nn_yaml

if __name__ == "__main__":

    # read in parameters from a yaml file
    # (eventually) have this be called as a scipt instead of
    # calling this file every time we want to run something

    # the yaml file is in the current directory, for now
    data_dict = parse_nn_yaml("options.yaml")
    nn_params = data_dict["Network"]

    batch_size = nn_params["options"]["batch_size"]
    in_size = nn_params["options"]["in_size"]
    out_size = nn_params["options"]["out_size"]

    @jax.jit
    @jax.grad
    def loss_fn(model, x, y):
        pred_y = jax.vmap(model)(x)
        return jax.numpy.mean((y - pred_y) ** 2)

    model = Linear(in_size, out_size, key=jax.random.PRNGKey(0))
    x = jax.numpy.zeros((batch_size, in_size))
    y = jax.numpy.zeros((batch_size, out_size))
    grads = loss_fn(model, x, y)

    # define the function we want to approximate, x^2 in this example:
    def f(x):
        return x ** 2

    # will want to make this more general in the future
    #def evaluate(model: Linear):
    #    # why is this array being converted toa numpy array for each evaluation?
    #    x = x.numpy()
    #    y = y.numpy()
    #
    #    return 

    # I'm having trouble connecting the CNN image processing example and the
    # example of a single-layer network defined in equinox; need to sit down and figure
    # out what's going on
    #value, grads = eqx.filter_value_and_grad(loss_fn)(model, x, y)
    #print(value, grads)

    # fix the optimizer for now; allow user to specify optimizer in the future
    LEARNING_RATE = 1e-4 # no clue what it should be for my given problem
    optim = optax.adamw(LEARNING_RATE)

    # define training for the NN
    #def train(model: Linear, )