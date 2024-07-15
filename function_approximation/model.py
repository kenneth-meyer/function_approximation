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