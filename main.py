import jax
import jax.numpy as jnp
import tensorflow as tf
import tensorflowjs as tfjs


resolution = 90
ray_params = {}


def rot(a):
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
    u_1 = jnp.cross(f, r)
    c = f*z
    i = c + u*r + v*u_1
    return normalize(i)


def get_color(x, y):
    return x*y

def ray_march(ro, rd):
    distance = 0.0
    for i in range(20):
        distance += get_dist(ro + rd * distance)
    return distance


# epsilon = 0.001
def ray_marchOpt(ro, rd):
    # vrai version, on tente d'optimiser?
    # distance = 0.0
    # for i in range(20):
    #     distance += get_dist(ro + rd * distance)
    # return distance


    # max_steps = 10
    #
    # distance = 0.0
    # add_distance = 1.0
    # i = 0
    # while jnp.where(jnp.logical_and(jnp.where(i < 30), jnp.where(add_distance > epsilon))):
    #     i += 1
    #     add_distance = get_dist(ro + rd * distance)
    #     distance += add_distance
    #
    # return distance


    # def f(carry, row):
    #
    #     even = 0
    #     for n in row:
    #         even += jax.lax.cond(n % 2 == 0, lambda: 1, lambda: 0)
    #
    #     return carry + even, even
    #
    # numbers = jnp.array([[3.0, 14.0], [15.0, -7.0], [16.0, -11.0]])
    # jax.lax.scan(f, 0, numbers)

    # def dist_acc(start_dist, item):
    #     acc_distance = get_dist(ro + rd * start_dist)
    #     return start_dist + acc_distance, acc_distance
    #
    # result_dist, _ = jax.lax.scan(dist_acc, 0.0, jnp.linspace(0.0, 0.0, num=20))
    # return result_dist

    # ça marche, mais pas d'augmentation de perf ça calcule ptet les deux
    epsilon = -0.001
    distance = 0.0
    early_stop = False
    for i in range(40):
        # idem Unsupported Ops in the model before optimization _SwitchN
        # distance_acc = jax.lax.switch(
        #     jnp.array(early_stop, int),
        #     [lambda: 0.0, lambda: get_dist(ro + rd * distance)]
        # )
        distance_acc = jnp.where(
            early_stop,
            0.0,
            get_dist(ro + rd * distance)
        )
        # Unsupported Ops in the model before optimization _SwitchN
        #   y'a switch mais pas SwitchN
        #     https://github.com/onnx/tensorflow-onnx/issues/2094 ONNX en veut pas
        #     mais jax.lax.switch fait du if?
        #   early stopping c'est nul sur les shaders? vu que ça va executer les autres branchecs en simd?
        # distance_acc = jax.lax.cond(
        #     early_stop,
        #     lambda: 0.0,
        #     lambda: get_dist(ro + rd * distance)
        # )
        early_stop = early_stop | (abs(distance_acc) < epsilon)
        # early_stop = jax.lax.cond(
        #     abs(distance_acc) < epsilon,
        #     lambda: True,
        #     lambda: early_stop
        # )
        distance += distance_acc

    return distance






def ray_color(u, v, xs):
    center = jnp.array([0.0, 0.0, 0.0])
    camera_init = jnp.array([3.0, 0.0, 0.0])
    sun = normalize(jnp.array([1.0, 2.0, 3.0]))

    camera = jnp.matmul(rot(xs/5), camera_init)

    rd = get_ray_dir(u, v, camera, center, 1.0)

    d = ray_march(camera, rd)
    at_surface = camera + rd * d
    normal = normalize(jax.grad(get_dist)(at_surface))

    # second pass, starting from reflection
    # reflection = d - 2*jnp.dot(normal, d)*normal
    # d_2 = ray_march(at_surface, reflection)
    # at_surface_2 = at_surface + reflection * d_2
    # normal_2 = normalize(jax.grad(get_dist)(at_surface_2))

    light = (jnp.dot(normal, sun) + 1) / 2
    # light = (jnp.dot(normal_2, sun) + 1) / 2

    # return jnp.remainder(light, 1.0)
    # return jnp.clip(light, 0.0, 1.0)
    return light


def main(ray_params, xs):
    cxr = jnp.linspace(-1.0, 1.0, num=resolution, endpoint=False)
    cyr = jnp.linspace(-1.0, 1.0, num=resolution, endpoint=False)

    ray_colors = jax.vmap(jax.vmap(ray_color, (0, None, None), 0), (None, 0, None), 1)

    return ray_colors(cxr, cyr, xs[0, 0])
    # return ray_colors(cxr, cyr, 0.0)

main(ray_params, jnp.array([[0.9]]))

tfjs.converters.convert_jax(
    main,
    ray_params,
    input_signatures=[tf.TensorSpec((1, 1), tf.float32)],
    model_dir='./'
)
