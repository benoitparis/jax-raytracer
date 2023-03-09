import jax
import jax.numpy as jnp
import tensorflow as tf
import tensorflowjs as tfjs


resolution = 92
ray_params = {
    'mouse_coords': jnp.array([0.5, 0.5]),
    'time': jnp.array([0])
}

def rot(a):
    s = jnp.sin(a)
    c = jnp.cos(a)
    return jnp.array(
        [[c, -s,  0],
         [s,  c,  0],
         [0,  0,  1]])

def cube_points(p, s):
    p_sec = jnp.absolute(p)-s
    return jnp.linalg.norm(p_sec) - 0.25


def get_dist(p):
    vec_1 = jnp.array([1.0, 1.0, 1.0])
    d = cube_points(p, vec_1)
    return d

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
    for i in range(10):
        distance += get_dist(ro + rd * distance)
    return distance


def ray_color(u, v, xs):
    center = jnp.array([0.0, 0.0, 0.0])
    ro_0 = jnp.array([3.0, 0.0, 0.0])
    sun = normalize(jnp.array([1.0, 2.0, 3.0]))

    #     // where is the mouse?
    #     vec2 m = iMouse.xy/iResolution.xy;

    # ro.yz *= Rot(-m.y*PI+1.);
    # ro.xz *= Rot(-m.x*TAU + iTime);

    ro = jnp.matmul(rot(xs), ro_0)

    # rd = get_ray_dir(u, v, ro, center, xs)
    rd = get_ray_dir(u, v, ro, center, 1.0)

    d = ray_march(ro, rd)
    at_surface = ro + rd * d
    normal = normalize(jax.grad(get_dist)(at_surface))

    # second pass
    # reflection = d - 2*jnp.dot(normal, d)*normal
    # d_2 = ray_march(at_surface, reflection)
    # at_surface_2 = at_surface + reflection * d_2
    # normal_2 = normalize(jax.grad(get_dist)(at_surface_2))

    light = jnp.dot(normal, sun)
    # light = jnp.dot(normal_2, sun)

    # on clip pour être dans la range; il faudra retirer
    return jnp.clip(light, 0.0, 1.0)
    # return jnp.clip(d, 0.0, 1.0)
    # return d


def main(ray_params, xs):
    print('#######')
    print(xs)
    print('#######')
    cxr = jnp.linspace(-1.0, 1.0, num=resolution, endpoint=False)
    cyr = jnp.linspace(-1.0, 1.0, num=resolution, endpoint=False)



    ray_colors = jax.vmap(jax.vmap(ray_color, (0, None, None), 0), (None, 0, None), 1)

    couter = ray_colors(cxr, cyr, xs[0])

# necessary?
    out = jnp.dstack((couter, couter, couter))
    print(out.shape)
    print(out)

    return out

main(ray_params, [0.9])

tfjs.converters.convert_jax(
    main,
    ray_params,
    input_signatures=[tf.TensorSpec((1), tf.float32)],
    model_dir='./'
)
