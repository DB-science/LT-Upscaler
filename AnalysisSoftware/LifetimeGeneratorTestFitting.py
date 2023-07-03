# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 14:21:31 2023

@author: Dominik Boras
"""
import random
random.seed(10)
print(random.random())
import numpy as np
from scipy.special import erf
from lmfit import Model
import matplotlib.pyplot as plt

def convolveAB(a,b):
    A = np.fft.fft(a)
    B = np.fft.fft(b)
    convAB = np.real(np.fft.ifft(A*B))
    return convAB

def LinearFitReconstruction(x,y):
    values = []
    for i in range(0,len(x)-1):
        values.append(y[i])
        n = int(((x[i+1]-x[i])/2.5)-1)
        diff = (y[i+1]-y[i])/n
        for n in range(1,n+1):
            point = y[i]+n*diff
            values.append(point)
    values.append(y[-1])
    return values
            
            
        

def SpectrumGenerator(spectrum_counts, numberOfBins,I1,Tau1,I2, Tau2 ):

    spectrum_counts= spectrum_counts
    numberOfBins = numberOfBins
    shift= 7.750
    BG_ratio = 0.04
    I1 = I1
    I2 = I2
    I3 = 0.172
    I4 = 0.003
    
    Tau1, Tau2, Tau3, Tau4 = Tau1, Tau2, 0.380, 2.750
    Lifetime_counts = spectrum_counts- spectrum_counts*BG_ratio
    
    I_faktor1 = round(I1* Lifetime_counts)
    I_faktor2 = round(I2* Lifetime_counts)
    I_faktor3 = round(I3* Lifetime_counts)
    I_faktor4 = round(I4* Lifetime_counts)
    BG_faktor = round(BG_ratio *spectrum_counts)
    
    Sum = I_faktor1 + I_faktor2 +I_faktor3+I_faktor4+BG_faktor
    #Gausian function for inaccuracy
    
   
    mu, sigma = 0, 0.099978
    mu1, sigma1 = 0, 0.097665
    
    
    IRF =np.random.normal(mu, sigma, 0.5*Sum)+np.random.normal(mu1, sigma1, 0.5*Sum)


    tau1 = shift + np.random.exponential(scale = Tau1,size = I_faktor1 )
    tau2 = shift + np.random.exponential(scale = Tau2, size = I_faktor2)
    tau3 = shift + np.random.exponential(scale = Tau3, size = I_faktor3)
    tau4 = shift + np.random.exponential(scale = Tau4, size = I_faktor4)

    BG_counts = np.random.uniform(low=0.000, high=50.000, size=BG_faktor)


    Exp = np.append(tau1 ,tau2)
    Exp = np.append(Exp ,tau3)
    Exp = np.append(Exp ,tau4)
    Exp1 = np.append(Exp ,BG_counts)

    Spectrum = np.add(Exp1,IRF)


     
    #Binning

    histogram = np.histogram(Spectrum, bins=numberOfBins, range=(0,50), weights=None)

    FinalSpectra = histogram[0]
    return FinalSpectra

def areaCalculation(x,y):
    area= []
    for i in range(0,x.size-1):
        area.append((x[i+1]-x[i])*y[i]+0.5*((x[i+1]-x[i])*(y[i+1]-y[i])))
    return area





    
    
def Fitting(x,y,WhichOne, Expected_Tau1, Expected_I1, Expected_Tau2,
            Expected_I2,Tau_source_1_fix,I_source_1_fix ,Tau_source_2_fix,
            I_source_2_fix,HowManyIRF, FWHM1, I_FWHM1 ,FWHM2, I_FWHM2):
    #guess of Background
    Begin = 0
    End = 100
    Data = y
    guess_BG =(sum(Data[Begin:End])/(End-Begin))
    
   
    if(WhichOne == 0):
        if HowManyIRF ==1:
            #4 component expontential distribution:
            def ExpDecay_4(x, I1, tau1, I2, tau2,I3, tau3, I4, tau4,numberOfCounts,FWHM  ,x0,args=()):  
                
        
                sigma = FWHM/(2*np.sqrt(2*np.log(2)))
                h=np.zeros(x.size)
                N=np.zeros(x.size)
                N=1.0/(sigma*np.sqrt(2*np.pi))
                yIRF=N*np.exp(-0.5*((x-x0)/sigma)**2);#
                yIRF = yIRF
                yIRF = yIRF/yIRF.sum()
                
    
                
                x1 = np.exp(-(x)/tau1)
                x1 = x1/x1.sum()
                h1 = convolveAB(x1*I1, yIRF)
                x2 = np.exp(-(x)/tau2)
                x2 = x2/x2.sum()
                h2 = convolveAB(x2*I2, yIRF)
                S1 = np.exp(-(x)/tau3)
                S1 = S1/S1.sum()
                h3 = convolveAB(S1*I3, yIRF)
                S2 = np.exp(-(x)/tau4)
                S2 = S2/S2.sum()
                h4 = convolveAB(S2*I4, yIRF)
                
                h = h1+h2+h3+h4
                h = h/sum(h)
                
               
                
                return  (h*numberOfCounts)+guess_BG
            

            FitModel_2ExpDecay = Model(ExpDecay_4)
            
           
            FitModel_2ExpDecay.set_param_hint('tau1', vary=True,min= 50)
            FitModel_2ExpDecay.set_param_hint('I1',vary=True, min= 0.0, max = 1.0 )
            FitModel_2ExpDecay.set_param_hint('I2',vary=True, min = 0.0, max = 1.0)
            FitModel_2ExpDecay.set_param_hint('tau2', vary=True,min=50)
            FitModel_2ExpDecay.set_param_hint('FWHM', vary=False,min=100., max= 300.)
            
            FitModel_2ExpDecay.set_param_hint('tau3', value = Tau_source_1_fix, vary = False)
            FitModel_2ExpDecay.set_param_hint('I3', value = I_source_1_fix, vary = False)
            FitModel_2ExpDecay.set_param_hint('tau4', value = Tau_source_2_fix, vary = False)
            FitModel_2ExpDecay.set_param_hint('I4', value = I_source_2_fix, vary = False)
            
            
            FitModel_2ExpDecay.set_param_hint('x0', min = 0.0, max= 50000)
            FitModel_2ExpDecay.set_param_hint('numberOfCounts', min = 100000)
            #
            
           
            fitWeightingSpec = np.ones(len(y))
            for i in range(len(y)):
                val = 1./np.sqrt(y[i]+1)#
                
                if np.isfinite(val): 
                    fitWeightingSpec[i] += val
                    
            """fitWeightingSpec = np.ones(len(y))
            
            
            for i in range(len(y)):
                val = 1./np.sqrt(y[i]+1)#
                
                if np.isfinite(val): 
                    fitWeightingSpec[i] += val
                else:
                    fitWeightingSpec[i] = y[i]"""
                    
                    
                    
            
            result = FitModel_2ExpDecay.fit(y, x=x,I1 =Expected_I1 ,tau1= Expected_Tau1, I2=Expected_I2, tau2 = Expected_Tau2, x0 = 7990,FWHM = FWHM1,  weights = fitWeightingSpec , method='leastsq',max_nfev=100000)
            
            t0 = (float)(result.params['x0'].value);
            FWHM = (float)(result.params['FWHM'].value);
            
    
            t1 = (float)(result.params['tau1'].value);
    
            t1_err = (result.params['tau1'].stderr);
            if t1_err is not None:
                    t1_err = float(t1_err)
            else:
                    t1_err = 0.1  # Assign a default value or handle it differentl
            t2 = (float)(result.params['tau2'].value);
            
            t2_err = (result.params['tau2'].stderr);
            if t2_err is not None:
                    t2_err = float(t2_err)
            else:
                    t2_err = 0.1  # Assign a default value or handle it differentl
            #yRes = (float)(result.params['y0'].value);
            #yRes_err = (float)(result.params['y0'].stderr);
    
            I1 = (float)(result.params['I1'].value);
            I1_err =(result.params['I1'].stderr);
            if I1_err is not None:
                    I1_err = float(I1_err)
            else:
                    I1_err = 0.1  # Assign a default value or handle it differentl
    
            I2 = (float)(result.params['I2'].value);
            I2_err =(result.params['I2'].stderr);
            if I2_err is not None:
                    I2_err = float(I2_err)
            else:
                    I2_err = 0.1  # Assign a default value or handle it differentl
            counts = (float)(result.params['numberOfCounts'].value);
            
            print("Convolutional Comp1:",t1,"+-",t1_err, I1,"+-",I1_err)#
            print("Convolutional Comp2:",t2,"+-",t2_err, I2,"+-",I2_err)#
    
            print("Convolutional Comp3:",Tau_source_1_fix, I_source_1_fix)
            print("Convolutional Comp4:",Tau_source_2_fix, I_source_2_fix)
    
            print("Convolutional t0:",t0)
            print("Convolutional FWHM:",FWHM)
            
            
            x0 = t0
            sigma = FWHM1/(2*np.sqrt(2*np.log(2)))
            N=np.zeros(x.size)
            N=1.0/(sigma*np.sqrt(2*np.pi))
            yIRF=N*np.exp(-0.5*((x-x0)/sigma)**2);
            yIRF_sum = yIRF.sum()
            yIRF = yIRF/yIRF_sum
    
            Comp1 = np.exp(-(x)/(t1))
            Comp1 = (Comp1/Comp1.sum())
            Comp1 = (convolveAB(Comp1, yIRF)*I1*counts)+guess_BG
    
            Comp2 = np.exp(-(x)/(t2))
            Comp2 = (Comp2/Comp2.sum())
            Comp2 = (convolveAB(Comp2, yIRF)*I2*counts)+guess_BG
    
            Comp3 = np.exp(-(x)/Tau_source_1_fix)
            Comp3 = (Comp3/Comp3.sum())
            Comp3 = (convolveAB(Comp3, yIRF)*I_source_1_fix*counts)+guess_BG
    
    
            Comp4 = np.exp(-(x)/Tau_source_2_fix)
            Comp4 = (Comp4/Comp4.sum())
            Comp4 = (convolveAB(Comp4, yIRF)*I_source_2_fix*counts)+guess_BG
            
            sigmaLevelResiduen = result.residual /(np.sqrt(result.best_fit))
            plt.semilogy(x,y,'b--')
    
            #plt.semilogy(xdata, result.init_fit, 'k--')
            plt.semilogy(x, result.best_fit, 'r-')
    
            plt.semilogy(x, Comp1, 'g-')
    
            plt.semilogy(x, Comp2, 'y-')
    
            plt.semilogy(x, Comp3, 'c-')
    
            plt.semilogy(x, Comp4, 'k-')
            plt.xlim(1540*5,2200*5)
            plt.show()
            plt.plot(sigmaLevelResiduen[1540:2200],'r--');
            plt.show()
            residuen_sum = sum(sigmaLevelResiduen[1540:2200])
            print('residuen_sum', residuen_sum)
            
        if HowManyIRF ==2:
            #4 component expontential distribution:
            def ExpDecay_4(x, I1, tau1, I2, tau2,I3, tau3, I4, tau4,numberOfCounts,x0, x1,args=()):  
                
                h=np.zeros(x.size)
        
                sigma = FWHM1/(2*np.sqrt(2*np.log(2))) 
                N=np.zeros(x.size)
                N=1.0/(sigma*np.sqrt(2*np.pi))
                yIRF1=N*np.exp(-0.5*((x-x0)/sigma)**2);#
                
                sigma2 = FWHM2/(2*np.sqrt(2*np.log(2)))
                N2=np.zeros(x.size)
                N2=1.0/(sigma2*np.sqrt(2*np.pi))
                yIRF2=N2*np.exp(-0.5*((x-x1)/sigma2)**2);
                
                yIRF1 = yIRF1/yIRF1.sum()
                
                yIRF2 = yIRF2/yIRF2.sum()
                
                yIRF = I_FWHM1*yIRF1 + I_FWHM1*yIRF2
                yIRF /=sum(yIRF)
                
                x1 = np.exp(-(x)/tau1)
                x1 = x1/x1.sum()
                x2 = np.exp(-(x)/tau2)
                x2 = x2/x2.sum()
                S1 = np.exp(-(x)/tau3)
                S1 = S1/S1.sum()
                S2 = np.exp(-(x)/tau4)
                S2 = S2/S2.sum()
                
                h = (I1*x1 + I2*x2 + I3*S1 + I4*S2)
                
                hConvIrf_norm = convolveAB(h, yIRF)
                
                return  (hConvIrf_norm*numberOfCounts)+guess_BG
            
            
            FitModel_2ExpDecay = Model(ExpDecay_4)
            FitModel_2ExpDecay.set_param_hint('tau1', vary=False,min= 0.00001, max = 1000.0)
            FitModel_2ExpDecay.set_param_hint('I1',vary=False, min= 0.0, max = 1.0)
            FitModel_2ExpDecay.set_param_hint('I2',vary=False, min = 0.0, max = 1.0)
            FitModel_2ExpDecay.set_param_hint('tau2', vary=False,min=0.000001, max = 1000.0)
            
            FitModel_2ExpDecay.set_param_hint('tau3', value = Tau_source_1_fix, vary = False)
            FitModel_2ExpDecay.set_param_hint('I3', value = I_source_1_fix, vary = False)
            FitModel_2ExpDecay.set_param_hint('tau4', value = Tau_source_2_fix, vary = False)
            FitModel_2ExpDecay.set_param_hint('I4', value = I_source_2_fix, vary = False)
            
         
            FitModel_2ExpDecay.set_param_hint('x0', min = 0.0, max = 50000)
            FitModel_2ExpDecay.set_param_hint('x1', min = 0.0, max = 50000)
            FitModel_2ExpDecay.set_param_hint('numberOfCounts', min = 100000)
            #
            
        
            
            
            fitWeightingSpec = np.ones(len(y))
            for i in range(len(y)):
                val = 1./np.sqrt(y[i]+1)#
                
                if np.isfinite(val): 
                    fitWeightingSpec[i] += val
                    

            
            result = FitModel_2ExpDecay.fit(y, x=x, I1 =Expected_I1 ,tau1= Expected_Tau1, I2=Expected_I2, tau2 = Expected_Tau2, x0 = 8020, x1 = 7990, weights = fitWeightingSpec , method='leastsq',max_nfev=100000)
            
            
            t0 = (float)(result.params['x0'].value);
            t0_1 = (float)(result.params['x1'].value);
    
            t1 = (float)(result.params['tau1'].value);
    
            #t1_err = (float)(result.params['tau1'].stderr);
    
            t2 = (float)(result.params['tau2'].value);
            #t2_err = (float)(result.params['tau2'].stderr);
    
            #yRes = (float)(result.params['y0'].value);
            #yRes_err = (float)(result.params['y0'].stderr);
    
            I1 = (float)(result.params['I1'].value);
            #I1_err = (float)(result.params['I1'].stderr);
    
            I2 = (float)(result.params['I2'].value);
            #I2_err = (float)(result.params['I2'].stderr);
            counts = (float)(result.params['numberOfCounts'].value);
            
            print("Convolutional Comp1:",t1, I1)#"+-",t1_err,,"+-",I1_err
            print("Convolutional Comp2:",t2, I2)#"+-",t2_err,,"+-",I2_err
    
            print("Convolutional Comp3:",Tau_source_1_fix, I_source_1_fix)
            print("Convolutional Comp4:",Tau_source_2_fix, I_source_2_fix)
    
            print("Convolutional t0:",t0)
            print("Convolutional t0_1:",t0_1)
        
            
            
            x0 = t0
            x1 = t0_1
            sigma = FWHM1/(2*np.sqrt(2*np.log(2))) 
            N=np.zeros(x.size)
            N=1.0/(sigma*np.sqrt(2*np.pi))
            yIRF1=N*np.exp(-0.5*((x-x0)/sigma)**2);#
            
            sigma2 = FWHM2/(2*np.sqrt(2*np.log(2)))
            N2=np.zeros(x.size)
            N2=1.0/(sigma2*np.sqrt(2*np.pi))
            yIRF2=N2*np.exp(-0.5*((x-x1)/sigma2)**2);
            
            yIRF1 = yIRF1/yIRF1.sum()
            
            yIRF2 = yIRF2/yIRF2.sum()
            
            yIRF = I_FWHM1*yIRF1 + I_FWHM1*yIRF2
    
            Comp1 = np.exp(-(x)/(t1))
            Comp1 = (Comp1/Comp1.sum())
            Comp1 = (convolveAB(Comp1, yIRF)*I1*counts)+guess_BG
    
            Comp2 = np.exp(-(x)/(t2))
            Comp2 = (Comp2/Comp2.sum())
            Comp2 = (convolveAB(Comp2, yIRF)*I2*counts)+guess_BG
    
            Comp3 = np.exp(-(x)/Tau_source_1_fix)
            Comp3 = (Comp3/Comp3.sum())
            Comp3 = (convolveAB(Comp3, yIRF)*I_source_1_fix*counts)+guess_BG
    
    
            Comp4 = np.exp(-(x)/Tau_source_2_fix)
            Comp4 = (Comp4/Comp4.sum())
            Comp4 = (convolveAB(Comp4, yIRF)*I_source_2_fix*counts)+guess_BG
            
            sigmaLevelResiduen = result.residual /(np.sqrt(result.best_fit))
            plt.semilogy(x,y,'b--')
    
            #plt.semilogy(xdata, result.init_fit, 'k--')
            plt.semilogy(x, result.best_fit, 'r-')
    
            plt.semilogy(x, Comp1, 'g-')
    
            plt.semilogy(x, Comp2, 'y-')
    
            plt.semilogy(x, Comp3, 'c-')
    
            plt.semilogy(x, Comp4, 'k-')
            plt.xlim(1570*5,2100*5)
            plt.show()
            plt.plot(sigmaLevelResiduen[1570:2100],'r--');
            plt.show()
            residuen_sum = sum(sigmaLevelResiduen[1570:2100])
            
            print('residuen_sum', residuen_sum)
    elif(WhichOne ==1):
        def Analytic_ExpDecay4(x, I1, tau1, I2, tau2,I3, tau3, I4, tau4,numberOfCounts  ,x0,args=()):
            valF1 = np.zeros(len(x))
            valF2 = np.zeros(len(x))
            valF3 = np.zeros(len(x))
            valF4 = np.zeros(len(x))
            
            Mu = x0
            FWHM1 = FWHM
            sigma = FWHM1/(2*np.sqrt(2*np.log(2)))
            
            tau1 = tau1
            tau2 = tau2
            
            for i in range(0,x.size-1):
                y_11 = np.exp(-(x[i]-Mu-(sigma*sigma)/(4*tau1))/tau1)*(1-erf((0.5*sigma/tau1)-(x[i]-Mu)/sigma))
                y_12 = np.exp(-(x[i+1]-Mu-(sigma*sigma)/(4*tau1))/tau1)*(1-erf((0.5*sigma/tau1)-(x[i+1]-Mu)/sigma))
                
                valF1[i] += 0.5*I1*(y_11-y_12-erf((x[i]-Mu)/sigma)+erf((x[i+1]-Mu)/sigma))
                
                y_21 = np.exp(-(x[i]-Mu-(sigma*sigma)/(4*tau2))/tau2)*(1-erf((0.5*sigma/tau2)-(x[i]-Mu)/sigma))
                y_22 = np.exp(-(x[i+1]-Mu-(sigma*sigma)/(4*tau2))/tau2)*(1-erf((0.5*sigma/tau2)-(x[i+1]-Mu)/sigma))
                
                valF2[i] += 0.5*I2*(y_21-y_22-erf((x[i]-Mu)/sigma)+erf((x[i+1]-Mu)/sigma))
                
                y_31 = np.exp(-(x[i]-Mu-(sigma*sigma)/(4*tau3))/tau3)*(1-erf((0.5*sigma/tau3)-(x[i]-Mu)/sigma))
                y_32 = np.exp(-(x[i+1]-Mu-(sigma*sigma)/(4*tau3))/tau3)*(1-erf((0.5*sigma/tau3)-(x[i+1]-Mu)/sigma))
                
                valF3[i] += 0.5*I3*(y_31-y_32-erf((x[i]-Mu)/sigma)+erf((x[i+1]-Mu)/sigma))
                
                y_41 = np.exp(-(x[i]-Mu-(sigma*sigma)/(4*tau4))/tau4)*(1-erf((0.5*sigma/tau4)-(x[i]-Mu)/sigma))
                y_42 = np.exp(-(x[i+1]-Mu-(sigma*sigma)/(4*tau4))/tau4)*(1-erf((0.5*sigma/tau4)-(x[i+1]-Mu)/sigma))
                
                valF4[i] += 0.5*I4*(y_41-y_42-erf((x[i]-Mu)/sigma)+erf((x[i+1]-Mu)/sigma))
                
            Final = valF1+valF2+valF3+valF4
            Final = (Final *numberOfCounts)+guess_BG
            return Final
        
        FitModel_2ExpDecay_analytic = Model(Analytic_ExpDecay4)
        FitModel_2ExpDecay_analytic.set_param_hint('tau1', min= 0.00001)
        FitModel_2ExpDecay_analytic.set_param_hint('I1', min= 0.0)
        #FitModel_2ExpDecay_analytic.set_param_hint('y0', min = 0.)
        FitModel_2ExpDecay_analytic.set_param_hint('x0', min = 0.0)
        FitModel_2ExpDecay_analytic.set_param_hint('I2', min = 0.0)
        FitModel_2ExpDecay_analytic.set_param_hint('tau2', vary=False, min=0.000001)
        FitModel_2ExpDecay_analytic.set_param_hint('tau3', value = Tau_source_1_fix, vary = False)
        FitModel_2ExpDecay_analytic.set_param_hint('I3', value = I_source_1_fix, vary = False)
        FitModel_2ExpDecay_analytic.set_param_hint('tau4', value = Tau_source_2_fix, vary = False)
        FitModel_2ExpDecay_analytic.set_param_hint('I4', value = I_source_2_fix, vary = False)
        FitModel_2ExpDecay_analytic.set_param_hint('numberOfCounts', min = 100000)
        
        fitWeightingSpec = np.ones(len(y))
    
        for i in range(len(y)):
            val = 1./np.sqrt(y[i]+1) 
            
            if np.isfinite(val): 
                fitWeightingSpec[i] += val
        
        result2 = FitModel_2ExpDecay_analytic.fit(y, x=x, I1 =Expected_I1 ,tau1= Expected_Tau1, I2=Expected_I2, tau2 = Expected_Tau2, x0 = 7970, weights = fitWeightingSpec , method='leastsq',max_nfev=300)
        t0_1 = (float)(result2.params['x0'].value);

        t1_1 = (float)(result2.params['tau1'].value);
        #t1_1err = (float)(result2.params['tau1'].stderr);

        t2_1 = (float)(result2.params['tau2'].value);
        #t2_1err = (float)(result2.params['tau2'].stderr);

        #yRes_1 = (float)(result2.params['y0'].value);
        #yRes_err = (float)(result.params['y0'].stderr);

        I1_1 = (float)(result2.params['I1'].value);
        #I1_1err = (float)(result2.params['ampl1'].stderr);

        #amplitude1_err = (float)(result.params['ampl1'].stderr);
        I2_1 = (float)(result2.params['I2'].value);
        #I2_1err = (float)(result2.params['ampl2'].stderr);
        
        counts = (float)(result2.params['numberOfCounts'].value);
        
        
        print("Analytic Comp1:",t1_1, I1_1)#,"+-", t1_1err,"+-", I1_1err
        print("Analytic Comp2:",t2_1, I2_1)#,"+-", t2_1err,"+-", I2_1err

        print("Analytic Comp3:",Tau_source_1_fix, I_source_1_fix)
        print("Analytic Comp4:",Tau_source_2_fix, I_source_2_fix)

        print("Analytic t0:",t0_1)  
        
        
        x0 = t0_1
        FWHM1 = FWHM
        sigma = FWHM1/(2*np.sqrt(2*np.log(2)))
        N=np.zeros(x.size)
        N=1.0/(sigma*np.sqrt(2*np.pi))
        yIRF=N*np.exp(-0.5*((x-x0)/sigma)**2);
        yIRF_sum = yIRF.sum()
        yIRF = yIRF/yIRF_sum

        Comp1 = np.exp(-(x)/(t1_1))
        Comp1 = (Comp1/Comp1.sum())
        Comp1 = (convolveAB(Comp1, yIRF)*I1_1*counts)+guess_BG

        Comp2 = np.exp(-(x)/(t2_1))
        Comp2 = (Comp2/Comp2.sum())
        Comp2 = (convolveAB(Comp2, yIRF)*I2_1*counts)+guess_BG

        Comp3 = np.exp(-(x)/Tau_source_1_fix)
        Comp3 = (Comp3/Comp3.sum())
        Comp3 = (convolveAB(Comp3, yIRF)*I_source_1_fix*counts)+guess_BG


        Comp4 = np.exp(-(x)/Tau_source_2_fix)
        Comp4 = (Comp4/Comp4.sum())
        Comp4 = (convolveAB(Comp4, yIRF)*I_source_2_fix*counts)+guess_BG
        

        sigmaLevelResiduen = result2.residual /(np.sqrt(result2.best_fit))
        plt.semilogy(x,y,'b--')

        #plt.semilogy(xdata, result.init_fit, 'k--')
        plt.semilogy(x, result2.best_fit, 'r-')

        plt.semilogy(x, Comp1, 'g-')

        plt.semilogy(x, Comp2, 'y-')

        plt.semilogy(x, Comp3, 'c-')

        plt.semilogy(x, Comp4, 'k-')
        plt.xlim(6000,30000)
        plt.show()
        plt.plot(sigmaLevelResiduen[1400:4000]);
        plt.show()
        
        
#function to use the Binning mask for no divided Spectra

def ConstructXdataPoints(x, Binning, BG_area):
    beforeSpectrumPoints = np.arange(0, BG_area*Binning, Binning, dtype = float)
    StartValue = beforeSpectrumPoints[-1] 
    for i in range (0,len(x)):
        newValue = (StartValue + (x[i]*Binning)/2)
        beforeSpectrumPoints = np.append(beforeSpectrumPoints, newValue)
        StartValue = StartValue + x[i]*Binning
    x_masked = beforeSpectrumPoints
    return x_masked

# function for masked x with divided Spectra

def ReconstructedXdataPoints(y,BG_area, Binning):
    ValuesWithoutBG = y[BG_area:-1]
    Y_BG = y[0:BG_area]
    n = 1
    ListOfBinnigMask = []
    #print(ValuesWithoutBG.size)
    #generate a Binning Mask for all equal values
    for i in range(0,ValuesWithoutBG.size-1):
        if( ValuesWithoutBG[i] == ValuesWithoutBG[i+1]):
            n+=1
        else:
            ListOfBinnigMask.append(n)
            n=1
    if (sum(ListOfBinnigMask)!=(ValuesWithoutBG.size)):
        LastBin = ValuesWithoutBG.size -sum(ListOfBinnigMask)-1
        ListOfBinnigMask.append(LastBin)
        
    ListOfBinnigMask = np.asarray(ListOfBinnigMask)        
    #print(ListOfBinnigMask.sum())
    #delete all y[i+1] if they equal y[i]
    Position = 0
    newY = np.zeros(ListOfBinnigMask.size)
    for i in range(0, ListOfBinnigMask.size):
        Position += ListOfBinnigMask[i]
        newY[i] = ValuesWithoutBG[Position]
    Y_masked = np.append(Y_BG, newY)
    
    
    return Y_masked, ListOfBinnigMask
        
