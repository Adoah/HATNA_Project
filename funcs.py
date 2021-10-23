import pandas as pd
import matplotlib.pyplot as plt
import models
#from penguins.class_sciData import sciData
import time
from numpy.random import random
from numpy.random import choice
import numpy as np
import scipy.optimize as sco

eV = 1 #Energy
K  = 1 #Temperature Units
C  = 1 #Coulombs
s  = 1 #seconds
V  = 1 #volts

kb = 8.6173324e-5*eV/K #Boltzmann Constant
q  = 1.6e-19*C
h  = 4.1356e-15*eV*s

def reducedLandauer(vb,n,gammaL,gammaR,deltaE1,eta,sigma):
    gammaC = gammaL*gammaR*n
    gammaW = gammaL+gammaR
    
    c = 0
    vg = 0
    T = 300
    return models.tunnelmodel_singleLevel(vb,n,gammaC,gammaW, deltaE1,
                                          eta,sigma,c,vg,T)
                                          
def reducedBridgePop(vb, gammaL, gammaR, deltaE, eta):
    c = 0
    vg = 0
    T = 300
    return models.averageBridgePopulation(vb, gammaL, gammaR, deltaE, eta, c, vg, T)

def MarcusETRates(vb, gamma, lam, E_AB):
    alpha = vb-E_AB
    T = 300*K
    S = 2*np.sqrt(np.pi*kb*T/lam)
    
    R_plus = (gamma/4)*S*np.exp(-(alpha+lam)**2/(4*lam*kb*T))
    R_minus = (gamma/4)*S*np.exp(-(alpha-lam)**2/(4*lam*kb*T))
    return R_plus,R_minus  

def NitzFit(V, n, gammaL, gammaR, kappa, sigma, E_AB, E_AC, chi, eta, gam, lam, P, u):
    
    eq1 = np.vectorize(reducedLandauer) 
    eq2 = np.vectorize(reducedBridgePop)
    eq3 = np.vectorize(MarcusETRates)
    
    I_S0 = n*eq1(V,n,gammaL, gammaR, E_AB, eta, sigma) 
    I_S1 = n*eq1(V,n, kappa*gammaL, kappa*gammaR, E_AB+chi, eta,sigma)
    #I_S0 = eq1(V,n,gammaL, gammaR, E_AB, eta, sigma) 
    #I_S1 = eq1(V,n, kappa*gammaL, kappa*gammaR, E_AB+chi, eta,sigma)
    
    
    
    n_S0 = eq2(V, gammaL, gammaR, E_AB, eta)
    n_S1 = eq2(V, kappa*gammaL, kappa*gammaR, E_AB+chi, eta)
    
    R_AC, R_CA = eq3(V, gam, lam, E_AC)
    R_BD, R_DB = eq3(V, kappa*gam, lam, E_AC+chi)
    
    k_S0_S1 = (1-n_S0)*R_AC + n_S0*R_BD
    k_S1_S0 = (1-n_S1)*R_CA + n_S1*R_DB
    
    delt = abs(V[0]-V[1])/u
    I = []
    Parray = []
    for i in range(len(V)):
        Parray += [P]
        I += [((1-P)*I_S0[i]+P*I_S1[i])]
        
        dPdt = k_S0_S1[i]-P*(k_S0_S1[i]+k_S1_S0[i])
        P = P+dPdt*delt
    return V, I, Parray

def saveParams(params,err,save):
    output = ''
    for nm in params.keys():
        output = output + '%.4e\t'%params[nm]
    output = output + '%.4e\n'%err
    f= open(save,"a")
    f.write(output)
    f.close()
