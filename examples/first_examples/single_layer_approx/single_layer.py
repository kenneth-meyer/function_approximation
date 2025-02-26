import jax
import optax
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PyTree  # https://github.com/google/jaxtyping

# functions in this repo
from function_approximation.model import Linear, NonLinear, P_1_FE, SingleLayer
from function_approximation.helpers import parse_nn_yaml
import function_approximation.model as models

if __name__ == "__main__":

    # read in parameters from a yaml file
    # (eventually) have this be called as a scipt instead of
    # calling this file every time we want to run something

    # the yaml file is in the current directory, for now
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
    
    #@jax.jit
    #@jax.grad
    def loss_fn(model, x, y) -> Float[Array, ""]:
        pred_y = jax.vmap(model)(x)
        return jax.numpy.mean((y - pred_y) ** 2)
    
    loss = eqx.filter_jit(loss_fn)  # JIT our loss function from earlier!

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

    # define model using getattr()
    try:
        # this might be slightly redundant
        model_class = getattr(models, model_type)
        model = model_class(jax.random.PRNGKey(0), **nn_params)
    except:
        raise Exception("The model you requested is not implemented,\
                            check function_approxmiation.models for existing NN classes")

    # # defines model
    # if model_type == "Linear":
    #     model = Linear(in_size, out_size, key=jax.random.PRNGKey(0))
    # elif model_type == "Nonlinear":
    #     model = NonLinear(in_size, out_size, key=jax.random.PRNGKey(0))
    # elif model_type == "P_1_FE":
    #     model = P_1_FE(in_size, out_size, jax.random.PRNGKey(0), h)
    # elif model_type == "SingleLayer":
    #     model = SingleLayer(in_size, out_size, layer_width, jax.random.PRNGKey(0))  
    # else:
    #     raise Exception("You're trying to use a model I haven't implemented yet")

    # define datapoints to test at
    #x = jnp.tile(jnp.linspace(0,1,batch_size), (in_size,1)).T
    #x = jnp.zeros((batch_size, in_size))


    x = jnp.linspace(0,1,batch_size)
    # turn into a (batch_size , in_size) array
    x = jnp.expand_dims(x,1)

    print(x.shape)

    # mildly confused to what batch_size and in_size
    # correspond to in this example; write/work this out!

    # doesn't need to be jit-compiled lol
    def f(x):
        # assumes x vector?...
        return x ** 2

    y = f(x)

    # computes gradients - # 7/31 - FAILS for SingleLayer case, 
    # where layers need to be unpacked from a list
    # is this defined correctly? need to check this out...
    #grads = grad_loss_fun(model, x, y)

    # print initial prediction to see what's going on
    print(jax.vmap(model)(x))
    print("^ NN predictions")

    # define the function we want to approximate, x^2 in this example:
    # does this need to be jit-compiled?
    
    # will want to vmap across first argument, should this happen inside the function?

    # define learning rate; should allow this to be specified/change
    LEARNING_RATE = 0.001
        
    num_epochs = 10000
    loss_data = jnp.zeros(num_epochs)

    ######### Training ############

    optim = optax.adamw(LEARNING_RATE)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def make_step(
        model: model_type,
        opt_state: PyTree,
        x: Float[Array, "batch"],
        y: Int[Array, " batch"],
    ):
        loss_value, grads = eqx.filter_value_and_grad(loss)(model, x, y)
        updates, opt_state = optim.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value

    # the lines below were taken from the CNN example; not really sure what they do
    # Loop over our training dataset as many times as we need.
    #def infinite_trainloader():
    #    while True:
    #        yield from trainloader

    # training loop
    for epoch in range(num_epochs):

        # here, we should draw x from a sample, and compute/identify y values


        # use the equinox value and grad function here
        #loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x, y)
        # (loss is value, right?)

        model, opt_state, train_loss = make_step(model, opt_state, x, y)

        # save the loss function at each step - doesn't need to be jax array..
        loss_data = loss_data.at[epoch].set(train_loss) # optimal?

    # save params to a file for postprocessing  
    # will need to update this to better/more strategically save results
    eqx.tree_serialise_leaves(results_dir + "/" + model_type + "_model.eqx", model)

    # save things to a file
    jnp.save(results_dir + model_type + "_loss", loss_data)