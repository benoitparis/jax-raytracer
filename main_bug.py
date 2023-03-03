import jax.numpy as jnp
import tensorflow as tf
import tensorflowjs as tfjs

params = {
    'mouse_coords': jnp.array([0.5, 0.5]),
    'time': jnp.array([0])
}
linx = jnp.linspace(0.0, 1.0, num=10, endpoint=False)
liny = jnp.linspace(0.0, 1.0, num=10, endpoint=False)
outer_outside = jnp.outer(linx, liny)

def main_outside(params, xs):
    out = jnp.dstack((outer_outside, outer_outside, outer_outside))
    print(out.shape)
    print(out)
    return out

main_outside(params, [0, 1])


def main_inside(params, xs):
    outer = jnp.outer(linx, liny)
    out = jnp.dstack((outer, outer, outer))
    print(out.shape)
    print(out)
    return out

main_inside(params, [0, 1])


tfjs.converters.convert_jax(
    main_outside,
    params,
    input_signatures=[tf.TensorSpec((2), tf.float32)],
    model_dir='./bug_outside/'
)

tfjs.converters.convert_jax(
    main_inside,
    params,
    input_signatures=[tf.TensorSpec((2), tf.float32)],
    model_dir='./bug_inside/'
)
