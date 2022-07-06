from matplotlib.colors import rgb2hex
import numpy as np

PALETTE = [
    [  0,   0,   0], # unlabeled     =   0,
    [ 70,  70,  70], # building      =   1,
    [100,  40,  40], # fence         =   2,
    [ 55,  90,  80], # other         =   3,
    [220,  20,  60], # pedestrian    =   4,
    [153, 153, 153], # pole          =   5,
    [157, 234,  50], # road line     =   6,
    [128,  64, 128], # road          =   7,
    [244,  35, 232], # sidewalk      =   8,
    [107, 142,  35], # vegetation    =   9,
    [  0,   0, 142], # vehicle       =  10,
    [102, 102, 156], # wall          =  11,
    [220, 220,   0], # traffic sign  =  12,
    [ 70, 130, 180], # sky           =  13,
    [ 81,   0,  81], # ground        =  14,
    [150, 100, 100], # bridge        =  15,
    [230, 150, 140], # rail track    =  16,
    [180, 165, 180], # guard rail    =  17,
    [250, 170,  30], # traffic light =  18,
    [110, 190, 160], # static        =  19,
    [170, 120,  50], # dynamic       =  20,
    [ 45,  60, 150], # water         =  21,
    [145, 170, 100], # terrain       =  22,
    [236, 236, 236], # general anomaly = 23,
]

nPal = [i[0] + 256 * i[1] + 256 * 256 * i[2] for i in PALETTE]

rbg2num = dict([tuple(reversed(i)) for i in enumerate(nPal)])
num2ood = {i: 0 for i in range(len(PALETTE))}