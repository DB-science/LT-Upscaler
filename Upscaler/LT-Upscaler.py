# -*- coding: utf-8 -*-
"""
Created on Tue May 16 10:28:44 2023

@author: Dominik Boras
"""
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
import h5py 
from sklearn.linear_model import LinearRegression
import LifetimeGeneratorTestFitting_nichtEquidistant as LTgenNoEqui
from scipy.interpolate import interp1d

def find_position(numbers, target):
    current_sum = 0

    for i, num in enumerate(numbers):
        current_sum += num

        if current_sum >= target:
            break

    return i


#How far do you want to predict
Prediction_distance = 11000
#How long do you wanna train the model
Train = 400
#Which Spectrum do you wanna predict ex. 5000 = 5Mio
Prediction = 10000
#How much Background Channels do you have on the left side
BG_channels = 1525
# do you want reconstructed Spectra ?
WantReconstructedSpectra = False

with h5py.File('Insitu_Alu5N_defomiert_200grad_1h_5Mio_zuH5datei.h5', 'r') as f1:
    print(list(f1.keys()))  # print list of root level objects
    # following assumes 'x' and 'y' are dataset objects
    ds_x1 = f1['Spectra']  # returns h5py dataset object for 'x'
    arr_x1 = f1['Spectra'][()]  # returns np.array for 'x'
    arr_x1 = ds_x1[()]  # uses dataset object to get np.array for 'x'
    print (arr_x1.shape)
df_Spectra = arr_x1

arr_x1 = np.delete(arr_x1, np.s_[0:3],0)

Channels = 10000


Spectras = arr_x1[:,0:Channels]#
#Spectras = Spectras.reshape((10001, 1,Channels)).sum(axis=1)
print(Spectras)

Sum_BG = np.sum(Spectras[:,0:BG_channels], axis=1)
Last_Input = Spectras[Train]
Max = np.amax(Last_Input, axis=0)
Max_loc = np.argmax(Last_Input)

X = []
for i in range(0,Train):
  X =np.append(X,i)
X = np.array(X)
X = X.reshape(-1,1)
X1 = []
for i in range(1,Prediction_distance):
  X1 =np.append(X1,i)
X1 = np.array(X1)
X1 = X1.reshape(-1,1)

# fit the model
BG = np.delete(Sum_BG, np.s_[Train:30001],0)
my_lr = LinearRegression()
my_lr.fit(X,BG)
# predict on the same period
BG_preds = my_lr.predict(X1)

BG_Prediction = BG_preds[Prediction]

BG_list =  []
a = 0
for k in range(0,BG_channels):
    a = BG_Prediction / BG_channels
    BG_list.append(a)

#automatic Binning


Spectras1 = arr_x1[:,BG_channels:Channels]
X = []
for i in range(0,Train):
  X =np.append(X,i)
X = np.array(X)
X = X.reshape(-1,1)
X1 = []
for i in range(1,Prediction_distance+1):
  X1 =np.append(X1,i)
X1 = np.array(X1)
X1 = X1.reshape(-1,1)
# fit the model
AllSteigung = []
AllRes = []
AllBins = []
SpectraBinned = []
res = 0
m = 0
n = 0
w = 0
Steigung = 0

Threshold = int(Max_loc -BG_channels)

for i in range(0,400):
    n = w
    BinnedSpectra=Spectras1[:,n+i:n+i+1].sum(axis=1)
    df_1 = np.delete(BinnedSpectra, np.s_[Train:30001],0)
    my_lr = LinearRegression()
    my_lr.fit(X,df_1)
    # predict on the same period
    preds = my_lr.predict(X1)
    Steigung =(preds[Train]-preds[0])/(Train)
    m = 0

    if (n+i) < Threshold :
        while Steigung<10.64:
            m +=1
            w +=1
            BinnedSpectra=Spectras1[:,n+i:n+i+1+m].sum(axis=1)
            df_1 = np.delete(BinnedSpectra, np.s_[Train:30001],0)
            my_lr.fit(X,df_1)
            # predict on the same period
            preds = my_lr.predict(X1)
            Steigung =(preds[Train]-preds[0])/(Train)
            if m == 2:
                break
                
    if (n+i) >= Threshold :
        while Steigung<20.64:
            m +=1
            w +=1
            BinnedSpectra=Spectras1[:,n+i:n+i+1+m].sum(axis=1)
            df_1 = np.delete(BinnedSpectra, np.s_[Train:30001],0)
            my_lr.fit(X,df_1)
            # predict on the same period
            preds = my_lr.predict(X1)
            Steigung =(preds[Train]-preds[0])/(Train)
            if m == 5:
                break
        if (n+i) >= Threshold :
            while Steigung<10.64:
                m +=1
                w +=1
                BinnedSpectra=Spectras1[:,n+i:n+i+1+m].sum(axis=1)
                df_1 = np.delete(BinnedSpectra, np.s_[Train:30001],0)
                my_lr.fit(X,df_1)
                # predict on the same period
                preds = my_lr.predict(X1)
                Steigung =(preds[Train]-preds[0])/(Train)
                if m == 10:
                    break
        if (n+i) >= Threshold :
            while Steigung<5.64:
                m +=1
                w +=1
                BinnedSpectra=Spectras1[:,n+i:n+i+1+m].sum(axis=1)
                df_1 = np.delete(BinnedSpectra, np.s_[Train:30001],0)
                my_lr.fit(X,df_1)
                # predict on the same period
                preds = my_lr.predict(X1)
                Steigung =(preds[Train]-preds[0])/(Train)
                if m == 30:
                    break
        while Steigung<0.22 :
            m +=1
            w +=1
            BinnedSpectra=Spectras1[:,n+i:n+i+1+m].sum(axis=1)
            df_1 = np.delete(BinnedSpectra, np.s_[Train:30001],0)
            my_lr.fit(X,df_1)
            # predict on the same period
            preds = my_lr.predict(X1)
            Steigung =(preds[Train]-preds[0])/(Train)
            if m == 150:
                break
    SpectraBinned.append(preds[10000])
    AllBins.append(m+1)
    AllRes.append(res)
    AllSteigung.append(Steigung)

  


if (WantReconstructedSpectra == True):
    ResultSpectra = []
    z = 0
    k = 0
    l = 0
    
    for k in range(0,len(AllBins)):
        for l in range(0,AllBins[k]):
            PartList = []
            z = SpectraBinned[k]/AllBins[k]
            PartList.append(z)
            ResultSpectra.extend(PartList)
        if len(ResultSpectra) == Channels-BG_channels:
                break

    BG_list.extend(ResultSpectra)
    Final_Spectrum = BG_list
    Final_Spectrum = Final_Spectrum[0:Channels]

    plt.semilogy(Final_Spectrum,'b.')
    
else:
    ResultSpectra = []
    z = 0
    k = 0

    for k in range(0,len(AllBins)):
        PartList = []
        z = SpectraBinned[k]/AllBins[k]
        PartList.append(z)
        ResultSpectra.extend(PartList)
        if len(ResultSpectra) == Channels-BG_channels:
            break  
    x_binned = AllBins
    BG_list.extend(ResultSpectra)
    Final_Spectrum = BG_list
    position = find_position(x_binned, (10000 - BG_channels))
    Final_Spectrum = Final_Spectrum[0:BG_channels+ position]
    x_binned = x_binned[0:BG_channels+ position]    
    
"""BG_value = str(Final_Spectrum[0])
Predicted_spectrum1 = np.round(Final_Spectrum, 0)

x1 = str(Train/2)
filename = 'RealMeasurements/Insitu_Alu_defomiert_250grad_1h_240tsd.txt'
        
np.savetxt(filename,Predicted_spectrum1,fmt='%0d',newline='\n',header='counts at channel-width 20ps BG-Value' + BG_value  +'')"""




x_data = LTgenNoEqui.ConstructXdataPoints(x_binned, 5, BG_channels )
x_data1 = x_data[0:len(Final_Spectrum)]
#plt.semilogy(x_data1, Final_Spectrum)
#plt.xlim(7000,20000)
#plt.show()



"""Lin_FinalSpectrum = LTgen.LinearFitReconstruction(x_data1, Final_Spectrum)

Lin_FinalSpectrum = Lin_FinalSpectrum[3040:-1]
x_new =np.arange(0,50000,2.5)
x_new = x_new[0:len(Lin_FinalSpectrum)]"""

"""data = np.column_stack((x_data1, Final_Spectrum))  # Stack x and y arrays as columns

# Save the data to a text file
np.savetxt('data.txt', data, delimiter=' ', header='x y', fmt='%.8f')"""

y = LTgenNoEqui.Fitting(x                   = x_data1,
                        y                   = Final_Spectrum,
                        WhichOne            = 0. ,  # 0 is for reconvolution, 1 is for the analytic solution
                        Expected_Tau1       = 100., # in ps
                        Expected_I1         = 0.50 ,
                        Expected_Tau2       = 240., # in ps
                        Expected_I2         = 0.40,
                        Tau_source_1_fix    = 373.1006 , # in ps
                        I_source_1_fix      = 0.1389,
                        Tau_source_2_fix    = 3255.1934, # in ps
                        I_source_2_fix      = 0.0060,
                        HowManyIRF          = 1, #max two
                        FWHM1               = 185.5, # in ps
                        I_FWHM1             = 0.8904,
                        FWHM2               = 240.,  # in ps
                        I_FWHM2             = 0.1096,
                        numberOfBins        = 10000.,
                        binWidth_in_ps      = 5.0)  # in ps

