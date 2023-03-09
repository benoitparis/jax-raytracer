# General imports
# import json
# import numpy as np
# import os
# import glob
# import string

#from IPython.core.display import display, HTML, Javascript
#import google.colab.html
#import google.colab.output
import jax
import jax.numpy as jnp
import tensorflow as tf
import tensorflowjs as tfjs


resolution = 92
ray_params = {
    'mouse_coords': jnp.array([0.5, 0.5]),
    'time': jnp.array([0])
}


def cube_points(p, s):
    p_sec = jnp.absolute(p)-s
    return jnp.linalg.norm(p_sec) - 0.125


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
    ds = get_dist(ro + rd*3)
    return ds
    # return jnp.linalg.norm(rd) # c'est rd qui fuck up?

#     // this is the distance
#     float dO=0.;
#
# for(int i=0; i<MAX_STEPS; i++) {
#                                // let's go in that direction
# vec3 p = ro + rd*dO;
# // let's advance with no collision with the strategy that we're safe the biggest sphere that fits between us and the nearest object
# // should get sorta-exponential convergence
# float dS = GetDist(p);
# dO += dS;
# // are we too far? are we too close?
#
# // TODO can we do early stopping and/or bypass remainder with JAX?
# if(dO>MAX_DIST || abs(dS)<SURF_DIST) break;
# }
#
# return dO;

def per_ray(u, v, xs):
    # print(xs)
    center = jnp.array([0.0, 0.0, 0.0])
    ro = jnp.array([1.5, 0.5, 0.5])

    # rd = get_ray_dir(u, v, ro, center, xs[1])
    rd = get_ray_dir(u, v, ro, center, xs[0])
    # rd = get_ray_dir(u, v, ro, center, 1.0)

    d = ray_march(ro, rd)

    # return u*v
    return d/5

#
# // let's travel until we hit a surface
# // how long to go?
# float d = RayMarch(ro, rd);
# // let's go at the surface
# vec3 p = ro + rd * d;
# // where is the surface's normal?
# vec3 n = GetNormal(p);
# // we reflect off the surface
# vec3 r = reflect(rd, n);
#
# // let's define where the light comes from
# vec3 sun = normalize(vec3(1,2,3));
# // are you at a close angle to the light? rgb are the same here
# vec3 color = vec3(dot(n, sun));
#
#
# fragColor = vec4(color, 1.0);


def main(ray_params, xs):
    print('#######')
    print(xs)
    print('#######')
    cxr = jnp.linspace(0.0, 1.0, num=resolution, endpoint=False)
    cyr = jnp.linspace(0.0, 1.0, num=resolution, endpoint=False)

    #     // where is the mouse?
    #     vec2 m = iMouse.xy/iResolution.xy;


    # ro.yz *= Rot(-m.y*PI+1.);
    # ro.xz *= Rot(-m.x*TAU + iTime);

    # v1get_color = jax.vmap(get_color, (0, 0), 0)
    # vget_color = jax.vmap(get_color, (0, 1), 1)
    # ça marche?
    vget_color = jax.vmap(jax.vmap(per_ray, (0, None, None), 0), (None, 0, None), 1)
    # vget_color = jax.vmap(jax.vmap(get_color, (0, None), 0), (None, 0), 1)


    couter = vget_color(cxr, cyr, xs)
    # couter = vget_color(cxr, cyr)
    # couter = jnp.outer(cxr, cyr)


    out = jnp.dstack((couter, couter, couter))
    # out = jnp.dstack((ctouter, ctouter, ctouter))
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
