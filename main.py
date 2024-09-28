import jax
import jax.numpy as jnp
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


def distance_function(position):
    return jnp.linalg.norm(jnp.mod(position, 4.0) - 2.0) - 0.5


def normalize(v):
    return v / jnp.linalg.norm(v)




def get_ray_direction(u, v, camera_direction):
    vertical = jnp.array([0.0, 0.0, 1.0])
    r = normalize(jnp.cross(vertical, camera_direction))
    u_1 = jnp.cross(camera_direction, r)
    i = camera_direction + u*u_1 + v*r
    return normalize(i)


def get_camera(txy):
    position = jnp.array([3.5, 0.0, 0.0])
    position = jnp.matmul(rotate_z(txy[0]/20 + txy[1]/resolution - 0.5), position)
    position = position + jnp.array([0.0, 0.0, -5 * (txy[2]/resolution - 0.5)] )
    direction = normalize(-position)
    return position, direction


def ray_color(u, v, txy):
    camera_position, camera_direction = get_camera(txy)

    ray_direction = get_ray_direction(u, v, camera_direction)

    distance = 0.0
    for _ in range(20):
        distance += distance_function(camera_position + ray_direction * distance)

    point_at_surface = camera_position + ray_direction * distance
    normal_at_surface = jax.grad(distance_function)(point_at_surface)

    sun_direction = normalize(jnp.array([1.0, 2.0, 3.0]))

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
