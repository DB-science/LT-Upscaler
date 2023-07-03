# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 13:50:05 2023


@author: Dominik Boras
"""
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
import h5py 
from sklearn.linear_model import LinearRegression
import LifetimeGeneratorTestFitting as LTgen
from scipy.interpolate import interp1d




data = np.loadtxt('Insitu_Alu5N_defomiert_275grad_1h_150tsd.txt', skiprows=1)  # Skip the header row

#x = data[:, 0]  # Extract the first column as x
y = data # Extract the second column as y

# Create a regular grid of equidistant x values
regular_x = np.linspace(0, 50000, num=10000)

# Create an interpolation function
#interpolator = interp1d(x, y, kind='quadratic')

# Interpolate y values onto the regular grid
#interpolated_y = interpolator(regular_x)


#after binning
selected_x = regular_x#[::0]
#selected_y = interpolated_y[::2]


start_channel = 0


selected_x = selected_x[start_channel:-1]
selected_y = y[start_channel:-1]

y = LTgen.Fitting(      x                   = selected_x,
                        y                   = selected_y,
                        WhichOne            = 0. ,  # 0 is for reconvolution, 1 is for the analytic solution
                        Expected_Tau1       = 135.4546, # in ps
                        Expected_I1         = 0.5450 ,
                        Expected_Tau2       = 400.5128 , # in ps
                        Expected_I2         = 0.3185,
                        Tau_source_1_fix    = 380.7371    , # in ps
                        I_source_1_fix      = 0.1304         ,
                        Tau_source_2_fix    = 3298.0369, # in ps
                        I_source_2_fix      = 0.0060 ,
                        HowManyIRF          = 1, #max two
                        FWHM1               = 180.79, # in ps
                        I_FWHM1             = 0.8904,
                        FWHM2               = 240.7460,  # in ps
                        I_FWHM2             = 0.1096) 
