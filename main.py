import jax
import jax.numpy as jnp
from jax import lax
from jax import nn
import tensorflow as tf
import tensorflowjs as tfjs


resolution = 512


def rotate_z(a):
    s = jnp.sin(a)
    c = jnp.cos(a)
    # A rotation matrix
    return jnp.array(
        [[c, -s,  0],
         [s,  c,  0],
         [0,  0,  1]])

def sphere(position, size):
    return jnp.linalg.norm(position) - size


def cube(position, size):
    return jnp.max(jnp.abs(position) - size)


def union(d1, d2, k=0.1):
    # smooth minimum, as min is not differentiable; with a root
    # k the typical distance over which to smooth
    k *= 2.0
    x = d2 - d1
    return 0.5 * (d1 + d2 - jnp.sqrt(x * x + k * k))


def distance_function(position, time):
#     return jnp.linalg.norm(jnp.mod(position, 4.0) - 2.0) - 0.5
#     return sphere(position + 2.0, 1.0) - 1.0
#     return cube(position + 2.0, 1.0) - 1.0
    return union(cube(position, 0.0) - 1.0, sphere(position + 2.0 *jnp.sin(time), 1.0)) # marche pas..
#     return jnp.max(position + 2.0) - 1.0


def normalize(v):
    return v / jnp.linalg.norm(v)


def get_ray_direction(u, v, camera_direction):
    vertical = jnp.array([0.0, 0.0, 1.0])
    r = normalize(jnp.cross(vertical, camera_direction))
    u_1 = jnp.cross(camera_direction, r)
    i = camera_direction + u*u_1 + v*r
    return normalize(i)


def get_camera(txy):
    position = jnp.array([6.0, 0.0, 0.0]) # TODO change init or angle to minimize
    position = jnp.matmul(rotate_z(5 * (txy[1]/resolution - 0.5)), position) # TODO make it follow more
    position = position + jnp.array([0.0, 0.0, -10 * (txy[2]/resolution - 0.5)] )
    direction = normalize(-position)
    return position, direction


sun_direction = normalize(jnp.array([1.0, 2.0, 3.0]))


def ray_color(u, v, txy):
    camera_position, camera_direction = get_camera(txy)
    ray_direction = get_ray_direction(u, v, camera_direction)
    distance = 0.0

    for _ in range(20):
        distance += distance_function(camera_position + ray_direction * distance, txy[0])

    point_at_surface = camera_position + ray_direction * distance
    normal_at_surface = jax.grad(distance_function)(point_at_surface, txy[0])


    fog = jnp.clip(jnp.exp(- distance * distance *  0.005), 0.0, 1.0)

    # TODO add specular? https://www.shadertoy.com/view/MsBGW1
    # Light diffusion
    return ((jnp.dot(normal_at_surface, sun_direction) + 1) / 2 ) * fog


def main(params, txy):
    u_range = jnp.linspace(-1.0, 1.0, num=resolution)
    v_range = jnp.linspace(-1.0, 1.0, num=resolution)

    ray_colors = jax.vmap(jax.vmap(ray_color, (0, None, None), 0), (None, 0, None), 1)

    return ray_colors(u_range, v_range, txy)


tfjs.converters.convert_jax(
    apply_fn=main,
    params={},
    input_signatures=[tf.TensorSpec([3], tf.float32)],
    model_dir='./'
)
