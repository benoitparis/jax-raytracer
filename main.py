# General imports
import json
import numpy as np
import os
import glob
import string

#from IPython.core.display import display, HTML, Javascript
#import google.colab.html
#import google.colab.output
import jax
import jax.numpy as jnp
import tensorflow as tf
import tensorflowjs as tfjs

print(1+1)

resolution = 256
ray_params = {
    'mouse_coords': jnp.array([0.5, 0.5]),
    'time': jnp.array([0])
}

cxr = jnp.linspace(0.5, 1.0, num=resolution, endpoint=False)
cyr = jnp.linspace(0.5, 1.0, num=resolution, endpoint=False)
couter = jnp.outer(cxr, cyr)
cstack = jnp.dstack((couter, couter, couter))

def main(ray_params, xs):
    xr = jnp.linspace(0.5, 1.0, num=resolution, endpoint=False)
    yr = jnp.linspace(0.5, 1.0, num=resolution, endpoint=False)
    simple = jnp.linspace(0.1, 1.0, num=resolution*resolution, endpoint=False)

    outer = jnp.outer(xr, yr)

    # out = jnp.dstack((outer, outer, outer))
    # out = jnp.ones(shape=(resolution, resolution, 3))
    # out = outer
    # out = simple.reshape((resolution, resolution))
    # out = simple # marche pas
    # out = jnp.linspace(0.1, 1.0, num=resolution*resolution, endpoint=True) # marche pas
    # out = jnp.linspace(0.5, 1.0, num=resolution*2, endpoint=False) # marche OK
    # out = jnp.linspace(0.5, 1.0, num=resolution*resolution, endpoint=False) # marche pas
    out = jnp.dstack((couter, couter, couter))
    print(xr.shape)
    print(yr.shape)
    print(out.shape)
    print(out)

    return out

main(ray_params, [])

tfjs.converters.convert_jax(
    main,
    ray_params,
    input_signatures=[tf.TensorSpec((2), tf.float32)],
    model_dir='./'
)
