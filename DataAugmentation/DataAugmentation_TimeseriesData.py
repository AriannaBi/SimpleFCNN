# T. T. Um et al., “Data augmentation of wearable sensor data for parkinson’s disease monitoring using convolutional neural networks,” in Proceedings of the 19th ACM International Conference on Multimodal Interaction, ser. ICMI 2017. New York, NY, USA: ACM, 2017, pp. 216–220.

# https://dl.acm.org/citation.cfm?id=3136817
#
# https://arxiv.org/abs/1706.00527

# @inproceedings{TerryUm_ICMI2017,
#  author = {Um, Terry T. and Pfister, Franz M. J. and Pichler, Daniel and Endo, Satoshi and Lang, Muriel and Hirche, Sandra and Fietzek, Urban and Kuli\'{c}, Dana},
#  title = {Data Augmentation of Wearable Sensor Data for Parkinson's Disease Monitoring Using Convolutional Neural Networks},
#  booktitle = {Proceedings of the 19th ACM International Conference on Multimodal Interaction},
#  series = {ICMI 2017},
#  year = {2017},
#  isbn = {978-1-4503-5543-8},
#  location = {Glasgow, UK},
#  pages = {216--220},
#  numpages = {5},
#  doi = {10.1145/3136755.3136817},
#  acmid = {3136817},
#  publisher = {ACM},
#  address = {New York, NY, USA},
#  keywords = {Parkinson\&\#39;s disease, convolutional neural networks, data augmentation, health monitoring, motor state detection, wearable sensor},
# }

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline  # for warping
from transforms3d.axangles import axangle2mat

class Augmentation_techniques:
    # ## 1. Jittering

    # #### Hyperparameters :  sigma = standard devitation (STD) of the noise
    def DA_Jitter(X, sigma=0.01):
        myNoise = np.random.normal(loc=0, scale=sigma, size=X.shape)
        return X + myNoise


    # ## 2. Scaling

    # #### Hyperparameters :  sigma = STD of the zoom-in/out factor
    def DA_Scaling(X, sigma=0.5):
        scalingFactor = np.random.normal(loc=1.0, scale=sigma, size=(1, X.shape[1]))  # shape=(1,3)
        myNoise = np.matmul(np.ones((X.shape[0], 1)), scalingFactor)
        return X * myNoise


    # ## 3. Magnitude Warping

    # #### Hyperparameters :  sigma = STD of the random knots for generating curves
    #
    # #### knot = # of knots for the random curves (complexity of the curves)

    # "Scaling" can be considered as "applying constant noise to the entire samples" whereas "Jittering" can be considered as "applying different noise to each sample".

    # "Magnitude Warping" can be considered as "applying smoothly-varing noise to the entire samples"


    ## This example using cubic splice is not the best approach to generate random curves.
    ## You can use other aprroaches, e.g., Gaussian process regression, Bezier curve, etc.
    def GenerateRandomCurves(X, sigma=0.2, knot=4):
        xx = (np.ones((X.shape[1], 1)) * (np.arange(0, X.shape[0], (X.shape[0] - 1) / (knot + 1)))).transpose()
        yy = np.random.normal(loc=1.0, scale=sigma, size=(knot + 2, X.shape[1]))
        x_range = np.arange(X.shape[0])
        random_curves = []
        for i in range(X.shape[-1]):
            cs = CubicSpline(xx[:, i], yy[:, i])
            random_curves.append(cs(x_range))
        return np.array(random_curves).transpose()


    def DA_MagWarp(X, sigma=0.2):
        return X * GenerateRandomCurves(X, sigma)


    # ## 4. Time Warping

    # #### Hyperparameters :  sigma = STD of the random knots for generating curves
    #
    # #### knot = # of knots for the random curves (complexity of the curves)

    def DistortTimesteps(X, sigma=0.2):
        tt = GenerateRandomCurves(X, sigma)  # Regard these samples aroun 1 as time intervals
        tt_cum = np.cumsum(tt, axis=0)  # Add intervals to make a cumulative graph
        # Make the last value to have X.shape[0]
        for i in range(X.shape[-1]):
            t_scale = (X.shape[0] - 1) / tt_cum[-1, i]
            tt_cum[:, i] = tt_cum[:, i] * t_scale
        return tt_cum


    def DA_TimeWarp(X, sigma=0.2):
        tt_new = DistortTimesteps(X, sigma)
        X_new = np.zeros(X.shape)
        x_range = np.arange(X.shape[0])
        for i in range(X.shape[-1]):
            X_new[:, i] = np.interp(x_range, tt_new[:, i], X[:, i])
        return X_new


    # ## 5. Permutation

    # #### Hyperparameters :  nPerm = # of segments to permute
    # #### minSegLength = allowable minimum length for each segment

    def DA_Permutation(X, nPerm=4, minSegLength=10):
        X_new = np.zeros(X.shape)
        idx = np.random.permutation(nPerm)
        bWhile = True
        while bWhile:
            segs = np.zeros(nPerm + 1, dtype=int)
            segs[1:-1] = np.sort(np.random.randint(minSegLength, X.shape[0] - minSegLength, nPerm - 1))
            segs[-1] = X.shape[0]
            if np.min(segs[1:] - segs[0:-1]) > minSegLength:
                bWhile = False
        pp = 0
        for ii in range(nPerm):
            x_temp = X[segs[idx[ii]]:segs[idx[ii] + 1], :]
            X_new[pp:pp + len(x_temp), :] = x_temp
            pp += len(x_temp)
        return X_new