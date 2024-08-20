import jax
import jax.numpy as jnp
import tensorflow as tf
import tensorflowjs as tfjs


resolution = 512


def rotate(a):
    s = jnp.sin(a)
    c = jnp.cos(a)
    # A rotation matrix
    return jnp.array(
        [[c, -s,  0],
         [s,  c,  0],
         [0,  0,  1]])


def cube_vertex_spheres(position):
    # One cube corner
    point = jnp.array([1.0, 1.0, 1.0])
    # Reflected across each axis, we get 8 corners
    points = jnp.absolute(position) - point
    # Sphere surface is at 0.25 from a corner
    return jnp.linalg.norm(points) - 0.25


def distance_function(p):
    return cube_vertex_spheres(p)


def normalize(v):
    return v / jnp.linalg.norm(v)


def get_ray_direction(u, v, camera_direction):
    vertical = jnp.array([0.0, 1.0, 0.0])
    r = normalize(jnp.cross(vertical, camera_direction))
    # TODO investigate gimbal lock here, by removing normalize:
    # r = jnp.cross(vertical, camera_direction)
    u_1 = jnp.cross(camera_direction, r)
    i = camera_direction + u*r + v*u_1
    return normalize(i)


def ray_march(ro, rd):
    distance = 0.0
    for _ in range(20):
        distance += distance_function(ro + rd * distance)
    return distance


def ray_color(u, v, t):
    camera_init = jnp.array([3.0, 0.0, 0.0])
    camera_position = jnp.matmul(rotate(t[0]/20), camera_init)
    camera_direction = normalize(-camera_position)

    ray_direction = get_ray_direction(u, v, camera_direction)

    distance = ray_march(camera_position, ray_direction)
    point_at_surface = camera_position + ray_direction * distance
    normal = jax.grad(distance_function)(point_at_surface)

    sun = normalize(jnp.array([1.0, 2.0, 3.0]))

    # Light diffusion
    light = (jnp.dot(normal, sun) + 1) / 2
    return light

    # Light reflections
    # reflection = distance - 2*jnp.dot(normal, distance)*normal
    # d_2 = ray_march(point_at_surface, reflection)
    # at_surface_2 = point_at_surface + reflection * d_2
    # normal_2 = normalize(jax.grad(distance_function)(at_surface_2))
    # light = (jnp.dot(normal_2, sun) + 1) / 2
    # return jnp.clip(light, 0.0, 1.0)


def main(params, t):
    cxr = jnp.linspace(-1.0, 1.0, num=resolution, endpoint=False)
    cyr = jnp.linspace(-1.0, 1.0, num=resolution, endpoint=False)

    ray_colors = jax.vmap(jax.vmap(ray_color, (0, None, None), 0), (None, 0, None), 1)

    return ray_colors(cxr, cyr, t)

# Debug with this
# main(ray_params, jnp.array([[0.9]]))

tfjs.converters.convert_jax(
    apply_fn=main,
    params={},
    input_signatures=[tf.TensorSpec([3], tf.float32)],
    model_dir='./'
)
