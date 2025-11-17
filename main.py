import jax
import jax.numpy as jnp
import tensorflow as tf
import tensorflowjs as tfjs


resolution = 512


def sphere(position, size):
    return jnp.linalg.norm(position) - size


def cube(position, size):
    return jnp.max(jnp.abs(position) - size)


def union(d1, d2, k=0.1):
    # smooth minimum, as min is not differentiable; root method
    # k the typical distance over which to smooth
    k *= 2.0
    x = d2 - d1
    return 0.5 * (d1 + d2 - jnp.sqrt(x * x + k * k))


def distance_function(position, time):
    return union(
        cube(position, 0.85),
        sphere(position + 2.0 * jnp.sin(time), 1.0)
    )


def normalize(v):
    return v / jnp.linalg.norm(v)



def get_ray_direction(u, v, camera_direction):
    vertical = jnp.array([0.0, 0.0, 1.0])
    r = normalize(jnp.cross(vertical, camera_direction))
    u_1 = jnp.cross(camera_direction, r)
    i = camera_direction + u*u_1 + v*r
    return normalize(i)


def rotate_z(a):
    s = jnp.sin(a)
    c = jnp.cos(a)
    # A rotation matrix
    return jnp.array(
        [[c, -s,  0],
         [s,  c,  0],
         [0,  0,  1]])


def get_camera(x, y):
    position = jnp.array([-2.0, 4.0, 0.0])
    position = jnp.matmul(rotate_z(5 * (x/resolution - 0.5)), position)
    position = position + jnp.array([0.0, 0.0, -10 * (y/resolution - 0.5)] )
    direction = normalize(-position)
    return position, direction


def reflect(incident, normal):
    return incident - 2 * jnp.dot(normal, incident) * normal


sun_direction = normalize(jnp.array([1.0, 2.0, 3.0]))


def ray_color(u, v, txy):
    time = txy[0]
    x = txy[1]
    y = txy[2]

    camera_position, camera_direction = get_camera(x, y)
    ray_direction = get_ray_direction(u, v, camera_direction)

    # Marching
    distance = 0.0
    for _ in range(20):
        distance += distance_function(camera_position + ray_direction * distance, time)

    point_at_surface = camera_position + ray_direction * distance

    normal_at_surface = jax.grad(distance_function)(point_at_surface, time)
    normal_at_surface = normalize(normal_at_surface)

    # Lighting
    # Fog
    fog = jnp.clip(jnp.exp(- distance * distance *  0.005), 0.0, 1.0)
    # Specular
    reflected = reflect(-sun_direction, normal_at_surface)
    specular = jnp.maximum(0.0, jnp.dot(ray_direction, reflected)) ** 20.0
    # Diffusion
    diffuse = (jnp.dot(normal_at_surface, sun_direction) + 1) / 2

    return (0.5 * diffuse + 0.5 * specular) * fog


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
