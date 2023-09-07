#author: Dominik Boras
import pandas as pd
import os
import numpy as np
from matplotlib import pyplot as plt
import h5py
from sklearn.linear_model import LinearRegression

#How far do you want to predict
Prediction_distance = 10000
#How long do you wanna train the model
Train = 300
#Which Spectrum do you wanna predict ex. 5000 = 5Mio
Prediction = 5000
#How much Background Channels do you have on the left side
BG_channels = 1470


with h5py.File('Alu_0.0325BG_158ps_0.825_380ps_0.172_2.75ns_0.003_235.4FWHM_alle1000Cts_5Mio_5ps_Binning_FastLFgen_160223.h5', 'r') as f1:
    print(list(f1.keys()))  # print list of root level objects
    # following assumes 'x' and 'y' are dataset objects
    ds_x1 = f1['Spectra']  # returns h5py dataset object for 'x'
    arr_x1 = f1['Spectra'][()]  # returns np.array for 'x'
    arr_x1 = ds_x1[()]  # uses dataset object to get np.array for 'x'
    print (arr_x1.shape)
df_Spectra = arr_x1

arr_x1 = np.delete(arr_x1, np.s_[0:1],0)
Channels = 10000
Max_Spec = np.amax(arr_x1, axis=1)
Sum_Spec = np.sum(arr_x1, axis=1)
Spectras = arr_x1[:,BG_channels:BG_channels+1500]
#Spectras = Spectras.reshape((10000, 1,Channels-BG_channels)).sum(axis=1)
print(Spectras.shape)


plt.semilogy(Spectras[4999][0:2500])


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
BinnedSpectra=np.array(Spectras[:,90:91])
df_1 = np.delete(BinnedSpectra, np.s_[Train:30001],0)
my_lr = LinearRegression()
my_lr.fit(X,df_1)
# predict on the same period
preds = my_lr.predict(X1)

plt.plot(preds[0:5000],'k-')
plt.plot(BinnedSpectra[0:5000],'g-')
plt.show()
preds[0]



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
n = 0
# fit the model
AllSteigung = []
AllRes = []
AllBins = []
res = 0
z = 0
w=0
for i in range(0,129):
    n=w
    BinnedSpectra=np.array(Spectras[:,n+i:n+i+1].sum(axis=1))#[:,n+i:n+i+1]
    df_1 = np.delete(BinnedSpectra, np.s_[Train:30001],0)
    my_lr = LinearRegression()
    my_lr.fit(X,df_1)
    # predict on the same period
    preds = my_lr.predict(X1)
    res = (preds[4999]-BinnedSpectra[4999])/(np.sqrt(BinnedSpectra[4999]))
    z +=1
    print(z)
    #Steigung = (preds[5000]-preds[10])/(4990)
    Steigung =(BinnedSpectra[Train]-BinnedSpectra[0])/(Train)
    m=1
    while (res<-2. or res>2.):
        BinnedSpectra=np.array(Spectras[:,n+i:n+i+(1+m)].sum(axis=1))#[:,n+i:i+(1+m)+n]
        #n +=1
        w +=1
        m +=1
        df_1 = np.delete(BinnedSpectra, np.s_[Train:30001],0)
        my_lr.fit(X,df_1)
        # predict on the same period
        preds = my_lr.predict(X1)
        res = (preds[4999]-BinnedSpectra[4999])/(np.sqrt(BinnedSpectra[4999])) 
        Steigung = (BinnedSpectra[Train]-BinnedSpectra[0])/(Train)
        if m==50:
            break
    print(Steigung)
    AllBins.append(m)
    AllRes.append(res)
    AllSteigung.append(Steigung)
print(AllBins)


histogramSteig = np.histogram(AllSteigung, bins=10000, range=(0,100), weights=None)
x_Steig = np.linspace(0, 100, 10000)
FinalSteig = histogramSteig[0]
Sum_Steig = np.sum(FinalSteig)
print(Sum_Steig)
print(FinalSteig)
s = 1
i=0
while i<(0.6826*Sum_Steig):
    i = np.sum(FinalSteig[0:s])
    s+= 1
print("Threshold_slope2 : ",s*0.01)

p = 1
o=0
while o<(0.95*Sum_Steig):
    o = np.sum(FinalSteig[0:p])
    p+= 1
print("Threshold_slope1 : ",p*0.01)      
histogramRes = np.histogram(AllRes, bins=40, range=(-6,6), weights=None)
x_Res = np.linspace(-4, 4, 40)
FinalRes = histogramRes[0]

histogramBins = np.histogram(AllBins, bins=60, range=(0,60), weights=None)
x_Bins = np.linspace(0, 60, 60)
FinalBins = histogramBins[0]

plt.plot(x_Steig,FinalSteig, 'b-')
plt.show()
plt.plot(x_Res,FinalRes, 'r-')
plt.show()
plt.plot(x_Bins,FinalBins, 'g-')
plt.show()


