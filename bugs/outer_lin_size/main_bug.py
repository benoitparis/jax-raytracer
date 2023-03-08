import jax.numpy as jnp
import tensorflowjs as tfjs

params = {}

def main_ok(params):
    lin = jnp.linspace(0.0, 1.0, num=159, endpoint=False)
    return jnp.outer(lin, lin)


def main_fail(params):
    lin = jnp.linspace(0.0, 1.0, num=160, endpoint=False)
    return jnp.outer(lin, lin)


tfjs.converters.convert_jax(
    main_ok,
    params,
    input_signatures=[],
    model_dir='./main_ok/'
)

tfjs.converters.convert_jax(
    main_fail,
    params,
    input_signatures=[],
    model_dir='./main_fail/'
)
