import sys
import matplotlib.pyplot as plt
import torch, random
import numpy as np

# Defining sampling distributions for different parameters for dream

# mixture of gaussians
def g_mix(mixtures, bounds = (-1., 1.)):
    while True:
        m, std = random.choices(
            mixtures,
            weights = [m[0] for m in mixtures],
            k= 1
        )[0][1:]

        v = (torch.randn(1).item() * std) + m

        if v >= bounds[0] and v <= bounds[1]:
            return v

# rotation about z axis
def samp_s3d_az(prec):
    return round(
        g_mix([
            (0.9, 0.0, 0.2),
            (0.05, 0.65, 0.1),
            (0.05, -0.65, 0.1),        
        ], (-1., 1.)),
        prec
    )

# rotation about y axis
def samp_s3d_ay(prec):
    return round(
        random.random() * 2 - 1,
        prec
    )

# rotation about x axis
def samp_s3d_ax(prec):
    return round(g_mix([
        (0.3, 0.25, 0.1),
        (0.2, -0.15, 0.1),
        (0.5, 0.0, .4)
    ], (-1., 1.)), prec)

# depth
def samp_s3d_d(prec):
    return round(g_mix([
            (0.8, 0.05, 0.05),
            (0.2, 0.6, 0.3),
        ], (0,1.5)
    ), prec)

# height
def samp_s3d_h(prec):
    return round(g_mix([
            (0.5, 0.1, 0.05),
            (0.4, 0.6, 0.15),
            (0.1, 0.25, 0.4),
        ], (0,1.25)
    ), prec)

# width
def samp_s3d_w(prec):
    return round(g_mix([
            (0.8, 0.05, 0.05),
            (0.2, 0.6, 0.3),
        ], (0, 1.4)
    ), prec)


# x axis loc
def samp_s3d_x(prec):
    return round(g_mix([
        (0.5, -0.35, 0.15),
        (0.5, 0.35, 0.15),
        (0.2, 0.0, 0.01)
    ]), prec)

# y axis loc
def samp_s3d_y(prec):
    return round(g_mix([
        (0.4, -0.25, 0.15),
        (0.5, 0.0, 0.01),
        (0.1, 0.25, 0.15)
    ]), prec)

# z axis loc
def samp_s3d_z(prec):
    return round(g_mix([
        (0.5, -0.35, 0.15),
        (0.5, 0.35, 0.15),
        (0.2, 0.0, 0.01)
    ]), prec)

# translational symmetry params
def samp_s3d_TransDist(prec):
    v = (random.random() * 1.4) + 0.1
    return round(v, prec)

# 3d angle, randomly choose axis
def samp_s3d_angle(prec):
    r = random.random()
    if r < 0.33:
        return samp_s3d_ax(prec)

    elif r < 0.66:
        return samp_s3d_ay(prec)
    
    else:
        return samp_s3d_az(prec)

