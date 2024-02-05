import jax.numpy as jnp
import tensorflow as tf
import tensorflowjs as tfjs

params = {}


def main(ray_params, xs):
    return jnp.linalg.norm(xs)


tfjs.converters.convert_jax(
    main,
    params,
    input_signatures=[tf.TensorSpec((1, 1), tf.float32)],
    model_dir='./bugs/input_bug/'
)
