import jax
import jax.numpy as jnp
import tensorflow as tf
import tensorflowjs as tfjs


resolution = 512
# TODO specify reflection option here?
ray_params = {}


def rotate(a):
    s = jnp.sin(a)
    c = jnp.cos(a)
    return jnp.array(
        [[c, -s,  0],
         [s,  c,  0],
         [0,  0,  1]])


def cube_vertex_spheres(pos):
    point = jnp.array([1.0, 1.0, 1.0])
    points = jnp.absolute(pos) - point
    return jnp.linalg.norm(points) - 0.25


def get_dist(p):
    return cube_vertex_spheres(p)


def normalize(v):
    return v / jnp.linalg.norm(v)


def get_ray_dir(u, v, p, l, z):
    f = normalize(l-p)
    vertical = jnp.array([0.0, 1.0, 0.0])
    r = normalize(jnp.cross(vertical, f))
    # investigate gimbal lock here, by removing normalize:
    # r = jnp.cross(vertical, f)
    u_1 = jnp.cross(f, r)
    c = f*z
    i = c + u*r + v*u_1
    return normalize(i)


def get_color(x, y):
    return x*y


def ray_march(ro, rd):
    distance = 0.0
    for _ in range(20):
        distance += get_dist(ro + rd * distance)
    return distance


def ray_color(u, v, xs):
    center = jnp.array([0.0, 0.0, 0.0])
    camera_init = jnp.array([3.0, 0.0, 0.0])
    sun = normalize(jnp.array([1.0, 2.0, 3.0]))

    camera = jnp.matmul(rotate(xs/5), camera_init)

    rd = get_ray_dir(u, v, camera, center, 1.0)

    d = ray_march(camera, rd)
    at_surface = camera + rd * d
    normal = jax.grad(get_dist)(at_surface)

    # No reflections
    light = (jnp.dot(normal, sun) + 1) / 2
    return light


    # Second pass, for reflections
    # reflection = d - 2*jnp.dot(normal, d)*normal
    # d_2 = ray_march(at_surface, reflection)
    # at_surface_2 = at_surface + reflection * d_2
    # normal_2 = normalize(jax.grad(get_dist)(at_surface_2))
    # light = (jnp.dot(normal_2, sun) + 1) / 2
    # return jnp.clip(light, 0.0, 1.0)


def main(ray_params, xs):
    cxr = jnp.linspace(-1.0, 1.0, num=resolution, endpoint=False)
    cyr = jnp.linspace(-1.0, 1.0, num=resolution, endpoint=False)

    ray_colors = jax.vmap(jax.vmap(ray_color, (0, None, None), 0), (None, 0, None), 1)

    return ray_colors(cxr, cyr, xs[0, 0])

# Debug with this
# main(ray_params, jnp.array([[0.9]]))

tfjs.converters.convert_jax(
    main,
    ray_params,
    input_signatures=[tf.TensorSpec((1, 1), tf.float32)],
    model_dir='./'
)
