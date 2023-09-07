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


#How far do you want to predict
Prediction_distance = 11000
#How long do you wanna train the model
Train = 300
#Which Spectrum do you wanna predict ex. 5000 = 5Mio
Prediction = 10000
#How much Background Channels do you have on the left side
BG_channels = 1500
#channel number
Channels = 10000



Threshold_slope1 = 20.64
Threshold_slope2 = 0.22

max_bin1 = 15
max_bin2 = 150


#load your insitu spectra
with h5py.File('RealMeasurements/Insitu_Alu5N_defomiert_600grad_8h_deformiert_1h90GradAusgeheilt_5Mio_zuH5datei.h5', 'r') as f1:
    print(list(f1.keys()))  # print list of root level objects
    # following assumes 'x' and 'y' are dataset objects
    ds_x1 = f1['Spectra']  # returns h5py dataset object for 'x'
    arr_x1 = f1['Spectra'][()]  # returns np.array for 'x'
    arr_x1 = ds_x1[()]  # uses dataset object to get np.array for 'x'
    print (arr_x1.shape)
df_Spectra = arr_x1

arr_x1 = np.delete(arr_x1, np.s_[0:3],0)




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
    """if (n+i) < Threshold :
        while Steigung<20.64:
            m +=1
            w +=1
            BinnedSpectra=Spectras1[:,n+i:n+i+1+m].sum(axis=1)
            df_1 = np.delete(BinnedSpectra, np.s_[Train:30001],0)
            my_lr.fit(X,df_1)
            # predict on the same period
            preds = my_lr.predict(X1)
            Steigung =(preds[Train]-preds[0])/(Train)
            if m == 2:
                break"""
                
    if (n+i) >= Threshold :
        while Steigung<Threshold_slope1:
            m +=1
            w +=1
            BinnedSpectra=Spectras1[:,n+i:n+i+1+m].sum(axis=1)
            df_1 = np.delete(BinnedSpectra, np.s_[Train:30001],0)
            my_lr.fit(X,df_1)
            # predict on the same period
            preds = my_lr.predict(X1)
            Steigung =(preds[Train]-preds[0])/(Train)
            if m == max_bin1:
                break
        while Steigung<Threshold_slope2 :
            m +=1
            w +=1
            BinnedSpectra=Spectras1[:,n+i:n+i+1+m].sum(axis=1)
            df_1 = np.delete(BinnedSpectra, np.s_[Train:30001],0)
            my_lr.fit(X,df_1)
            # predict on the same period
            preds = my_lr.predict(X1)
            Steigung =(preds[Train]-preds[0])/(Train)
            if m == max_bin2:
                break
    SpectraBinned.append(preds[Prediction])
    AllBins.append(m+1)
    AllRes.append(res)
    AllSteigung.append(Steigung)


histogramSteig = np.histogram(AllSteigung, bins=100, range=(0,100), weights=None)
x_Steig = np.linspace(0, 100, 100)
FinalSteig = histogramSteig[0]
histogramRes = np.histogram(AllRes, bins=40, range=(-6,6), weights=None)
x_Res = np.linspace(-4, 4, 40)
FinalRes = histogramRes[0]

histogramBins = np.histogram(AllBins, bins=100, range=(0,100), weights=None)
x_Bins = np.linspace(0, 100, 100)
FinalBins = histogramBins[0]


Spectrum = Spectras[297]
Binned_real_spectrum = []
BG_list_real = []
Sum_real_BG = np.sum(Spectrum[0:BG_channels])

w = 0
k = 0
for k in range(0,BG_channels):
    w = Sum_real_BG / BG_channels
    BG_list_real.append(w)

Spectrum1 = Spectrum[BG_channels:Channels]

r = 0
e = 0
t = 0
u = 0
for e in range(0,len(AllBins)):
    for r in range(0,AllBins[e]):
        Part_binned =[]
        t = (np.sum(Spectrum1[u:u + AllBins[e]]))/AllBins[e]
        Part_binned.append(t)
        Binned_real_spectrum.extend(Part_binned)
    u += AllBins[e]


BG_list_real.extend(Binned_real_spectrum)
Final_Real_Spectrum_binned = BG_list_real
print(len(Final_Real_Spectrum_binned))
Final_Real_Spectrum_binned = Final_Real_Spectrum_binned[0:Channels]
plt.semilogy(Final_Real_Spectrum_binned, 'r.')


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
print(len(Final_Spectrum))

BG_value = str(Final_Spectrum[0])
Predicted_spectrum1 = np.round(Final_Spectrum, 0)

x1 = str(Train/2)
filename = 'RealMeasurements/Alu5N_defomiert_600grad_8h_deformiert_1h90GradAusgeheil.txt'
        
np.savetxt(filename,Predicted_spectrum1,fmt='%0d',newline='\n',header='counts at channel-width 20ps BG-Value' + BG_value  +'')
