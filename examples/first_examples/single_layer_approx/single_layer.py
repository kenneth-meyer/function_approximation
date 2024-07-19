import jax
import optax
import equinox as eqx
import jax.numpy as jnp

# functions in this repo
from function_approximation.model import Linear, NonLinear, P_1_FE
from function_approximation.helpers import parse_nn_yaml

if __name__ == "__main__":

    # read in parameters from a yaml file
    # (eventually) have this be called as a scipt instead of
    # calling this file every time we want to run something

    # the yaml file is in the current directory, for now
    data_dict = parse_nn_yaml("examples/first_examples/single_layer_approx/options.yaml")
    nn_params = data_dict["Network"]

    # Linear, Nonlinear, and P_1_FE 
    model_type = nn_params["name"]

    batch_size = nn_params["options"]["batch_size"]
    in_size = nn_params["options"]["in_size"]
    out_size = nn_params["options"]["out_size"]

    # we're working in x \in [0,1], so we can determine step size for FE mimicing
    h = 2.0 / float(batch_size)
    # I'll need to better check what FE we can/can't represent

    @jax.jit
    #@jax.grad
    def loss_fn(model, x, y):
        pred_y = jax.vmap(model)(x)
        return jax.numpy.mean((y - pred_y) ** 2)

    # separates the loss and gradient computation, but is slightly wasteful
    @jax.jit
    @jax.grad
    def grad_loss_fun(model,x,y):
        return loss_fn(model,x,y)

    # get loss and gradient in one step
    #@eqx.filter_value_and_grad
    #def loss_fn(model, x, y):
    #    pred_y = jax.vmap(model)(x)
    #    return jax.numpy.mean((y - pred_y) ** 2)


    # evaluate the model - uneccesarily computes pred_y, will be twice as expensive
    @jax.jit
    def eval(model, x):
        pred_y = jax.vmap(model)(x)
        return pred_y

    # defines model
    if model_type == "Linear":
        model = Linear(in_size, out_size, key=jax.random.PRNGKey(0))
    elif model_type == "Nonlinear":
        model = NonLinear(in_size, out_size, key=jax.random.PRNGKey(0))
    elif model_type == "P_1_FE":
        model = P_1_FE(in_size, out_size, jax.random.PRNGKey(0), h)
    else:
        raise Exception("You're trying to use a model I haven't implemented yet")

    # define datapoints to test at
    x = jnp.tile(jnp.linspace(0,1,batch_size), (in_size,1)).T

    print(x)

    # mildly confused to what batch_size and in_size
    # correspond to in this example; write/work this out!

    # doesn't need to be jit-compiled lol
    def f(x):
        # assumes x vector?...
        return x ** 2

    y = f(x)

    # computes gradients
    # is this defined correctly? need to check this out...
    grads = grad_loss_fun(model, x, y)

    # print initial prediction to see what's going on
    print(jax.vmap(model)(x))
    print("^ NN predictions")

    # define the function we want to approximate, x^2 in this example:
    # does this need to be jit-compiled?
    
    # will want to vmap across first argument, should this happen inside the function?

    # perform gradient descent once (does this do one iteration? idk what this is )
    LEARNING_RATE = 0.1
    new_model = model 
        
    num_epochs = 10000
    loss_data = jnp.zeros(num_epochs)

    # training loop
    for epoch in range(num_epochs):

        # reassign new_model to model - not sure if this something that is handled
        # well by jax or not
        #model = new_model

        # update y - use same x to sample
        #y_i = eval(new_model, x)

        # get gradients and loss itself
        loss = loss_fn(new_model, x, y)
        grads = grad_loss_fun(new_model,x,y)

        # save the loss function at each step - doesn't need to be jax array..
        loss_data = loss_data.at[epoch].set(loss) # optimal?

        # update params/the model - can we edit without calling tree_util?
        new_model = jax.tree_util.tree_map(lambda m, g: m - LEARNING_RATE * g, model, grads)


    # save the parameters to a file so I can do postprocessing separately
    results_dir = "/home/kenneth/research/learning/ML/function_approximation_data/" # hardcoded for now, want to make sure equinox works well
    eqx.tree_serialise_leaves(results_dir + "/" + model_type + "_model.eqx", new_model)
    # "model" is just the previous new_model I think, so results likely won't change
    # very much...

    print("saved to: ", results_dir)
    print(loss_data)

    # try to see if the model works, quickly
    print(jax.vmap(new_model)(x))
    print("^ NN predictions")

    print(f(x))
    print("^ actual data")


    jnp.save(results_dir + model_type + "_loss", loss_data)