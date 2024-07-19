import equinox as eqx
import jax

class Linear(eqx.Module):
    weight: jax.Array
    bias: jax.Array

    def __init__(self, in_size, out_size, key):
        wkey, bkey = jax.random.split(key)
        self.weight = jax.random.normal(wkey, (out_size, in_size))
        self.bias = jax.random.normal(bkey, (out_size,))

    def __call__(self,x):
        """ Determines action when the class is called

        Allows Linear class to be called directly to compute
        
        $$\text{NN}(x) = Wx + b, \quad W \in \mathcal{R}^{m \times n}, x \in \mathcal{R}^{m}, b \in \mathcal{R}^{m}$$

        Parameters
        ----------
        x : jax.Array
            input to the NN

        Returns
        -------
        W @ x + b
            output layer of a NN with a single layer
        """
        return self.weight @ x + self.bias
    
class NonLinear(eqx.Module):
    weight: jax.Array
    bias: jax.Array

    def __init__(self, in_size, out_size, key):
        wkey, bkey = jax.random.split(key)
        self.weight = jax.random.normal(wkey, (out_size, in_size))
        self.bias = jax.random.normal(bkey, (out_size,))

    def __call__(self,x):
        """ Determines action when the class is called

        Allows NonLinear class to be called directly to compute
        <insert thing here>

        Parameters
        ----------
        x : jax.Array
            input to the NN

        Returns
        -------
        
            output layer of a NN with a single layer and nonlinear activation function
        """

        return jax.nn.sigmoid(self.weight @ x + self.bias)

class P_1_FE(eqx.Module):
    weight: jax.Array
    bias: jax.Array
    h: float

    def __init__(self, in_size, out_size, key, h_val):
        wkey, bkey = jax.random.split(key)
        # need to update these depending on the domain we're approximating x on
        # and for the activation function we're using
        self.weight = jax.random.uniform(wkey, (out_size, in_size))
        self.bias = jax.random.uniform(bkey, (out_size,))

        # h is a step size determined from the spacing of x, which is a
        # function of the batch size in this case
        self.h = h_val

    def __call__(self,x):
        """ Determines action when the class is called

        Allows NonLinear class to be called directly to compute
        <insert thing here>

        Parameters
        ----------
        x : jax.Array
            input to the NN

        Returns
        -------
        
            output layer of a NN with a single layer and nonlinear activation function
        """

        # ignore bias
        return jax.nn.relu(self.weight @ (x + self.h)) - 2 * jax.nn.relu(self.weight @ x) + jax.nn.relu(self.weight @ (x - self.h))
    

class Single_layer(eqx.Module):
    weight: jax.Array
    bias: jax.Array

    def __init__(self, in_size, out_size, layer_width, key):
        wkey, bkey = jax.random.split(key)
        self.weight = jax.random.normal(wkey, (out_size, in_size))
        self.bias = jax.random.normal(bkey, (out_size,))
        self.layer_width = layer_width

    def __call__(self,x):
        """ Determines action when the class is called

        Allows NonLinear class to be called directly to compute
        <insert thing here>

        Parameters
        ----------
        x : jax.Array
            input to the NN

        Returns
        -------
        
            output layer of a NN with a single layer and nonlinear activation function
        """

        return jax.nn.sigmoid(self.weight @ x + self.bias)
