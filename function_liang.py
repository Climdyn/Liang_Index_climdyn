 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Function to compute the transfer of information from one variable x2 to the other x1 (T21) and the corresponding normalization (tau21)
T21: Liang (2014 eq. (10)), https://journals.aps.org/pre/abstract/10.1103/PhysRevE.90.052150
tau21: Liang (2015; eqs. (16-21)), https://doi.org/10.1103/PhysRevE.92.022126

Also compute Pearson correlation coefficient R

Compute the error in T21, tau21 and R based on the normal bootstrap with replacement (resampling of variables)

Inputs: 
x1: variable 1
x2: variable 2
dt: time step (e.g. 1 month)
n_iter: number of bootstrap realizations for computing the error

Outputs:
T21: absolute rate of information transfer from x2 to x1
tau21: relative rate of information transfer from x2 to x1
error_T21: error in T21 based on bootstrap resampling
error_tau21: error in tau21 based on bootstrap resampling
R: Pearson correlation coefficient
error_R: error in R based on bootstrap resampling
error_T21_FI based on Fisher information matrix

Created: 09/04/2021
Last updated: 23/11/2021

@author: David Docquier
"""

import numpy as np
from sklearn.utils import resample

def compute_liang(x1,x2,dt,n_iter):
    
    # Function to compute transfer of information from x2 to x1 (absolute: T21; and relative: tau21)
    def compute_liang_index(C11,C12,C22,C1d1,C2d1,Cd1d1,detC,dt):
        T21 = (C11 * C12 * C2d1 - C12**2 * C1d1) / (C11**2 * C22 - C11 * C12**2) # absolute rate of information flowing from x2 to x1 (nats per unit time); eq. (10) of Liang (2014)
        p = (C22 * C1d1 - C12 * C2d1) / detC # eq. (16) of Liang (2015)
        q = (-C12 * C1d1 + C11 * C2d1) / detC # eq. (17) of Liang (2015)
        expansion = p # phase-space expansion along x1; eq. (18) of Liang (2015)
        noise = (dt / (2 * C11)) * (Cd1d1 + p**2. * C11 + q**2. * C22 - 2. * p * C1d1 - 2. * q * C2d1 + 2. * p * q * C12) # stochastic process; eq. (19) of Liang (2015)
        Z21 = np.abs(T21) + np.abs(expansion) + np.abs(noise) # normalizer; eq. (20) of Liang (2015)
        tau21 = 100. * T21 / Z21 # relative rate of information from x2 to x1 (%); eq. (21) of Liang (2015)
        R = C12 / np.sqrt(C11 * C22) # compute correlation coefficient between x1 and x2; eq. (7) of Liang (2014)
        return T21,tau21,R
    
    # Compute tendency dx1
    k = 1 # k = 1 (or 2 for highly chaotic and densely sampled systems)
    N = np.size(x1) # length of the time series (number of observations)
    dx1 = np.zeros(N) # initialization of dx1 (to have the same number of time steps as x1)
    dx1[0:N-k] = (x1[k:N] - x1[0:N-k]) / (k * dt) # Euler forward finite difference of x1; eq. (7) of Liang (2014)
    
    # Compute covariances and matrix determinant
    C = np.cov(x1,x2) # covariance matrix between x1 and x2
    C11 = C[0,0] # variance of x1
    C12 = C[0,1] # covariance between x1 and x2
    C22 = C[1,1] # variance of x2
    C1d1 = np.cov(x1,dx1)[0,1] # covariance between x1 and dx1
    C2d1 = np.cov(x2,dx1)[0,1] # covariance between x2 and dx1
    Cd1d1 = np.var(dx1) # variance of dx1
    detC = C11 * C22 - C12**2. # determinant of covariance matrix
    
    # Compute absolute and relative transfers of information, and correlation coefficient
    T21,tau21,R = compute_liang_index(C11,C12,C22,C1d1,C2d1,Cd1d1,detC,dt)
    
    # Resample variables using bootstrap with replacement and compute corresponding T21, tau21 and R
    boot_T21 = np.zeros(n_iter)
    boot_tau21 = np.zeros(n_iter)
    boot_R = np.zeros(n_iter)
    for i in np.arange(n_iter): # loop over realizations
        
        # Resample x1, x2 and dx1
        boot_x1,boot_x2,boot_dx1 = resample(x1,x2,dx1,replace=True) # keep the same index for the 3 variables for the resampling
            
        # Compute covariances and matrix determinant based on resampled variables
        boot_C = np.cov(boot_x1,boot_x2)
        boot_C11 = boot_C[0,0]
        boot_C12 = boot_C[0,1]
        boot_C22 = boot_C[1,1]
        boot_C1d1 = np.cov(boot_x1,boot_dx1)[0,1]
        boot_C2d1 = np.cov(boot_x2,boot_dx1)[0,1]
        boot_Cd1d1 = np.var(boot_dx1)
        boot_detC = boot_C11 * boot_C22 - boot_C12**2.
        
        # Compute bootstrapped T21 and tau21, and correlation coefficient R
        boot_T21[i],boot_tau21[i],boot_R[i] = compute_liang_index(boot_C11,boot_C12,boot_C22,boot_C1d1,boot_C2d1,boot_Cd1d1,boot_detC,dt)
    
    # Compute error based on boostraped values (standard deviation)
    error_T21 = np.nanstd(boot_T21)
    error_tau21 = np.nanstd(boot_tau21)
    error_R = np.nanstd(boot_R)
    
    # Compute error based on the Fisher information matrix - based on page 4 of Liang (2014) and Matlab program of Liang
    a11 = (C22 * C1d1 - C12 * C2d1) / detC
    a12 = (-C12 * C1d1 + C11 * C2d1) / detC
    f1 = np.nanmean(dx1) - a11 * np.nanmean(x1) - a12 * np.nanmean(x2)
    R1 = dx1 - (f1 + a11 * x1 + a12 * x2) # eq. (8) of Liang (2014)
    Q1 = np.sum(R1**2.)
    b1 = np.sqrt(Q1 * dt / N) # eq. (9) of Liang (2014)
    nvar = 2
    x = np.column_stack((x1,x2)) # create a 2D matrix with x1 and x2 in 2 different columns
    NI = np.zeros([nvar+2,nvar+2]) # I: Fisher information matrix; N: length of time series
    NI[0,0] = N * dt / b1**2. # 1st element of 1st row of I
    NI[nvar+1,nvar+1] = 3. * dt / b1**4. * np.sum(R1**2.) - N / b1**2. # last element of last row of I
    for k in np.arange(nvar):
        NI[0,k+1] = dt / b1**2. * np.sum(x[:,k]) # 1st row of I
    NI[0,nvar+1] = 2. * dt / b1**3. * np.sum(R1) # last element of 1st row of I
    for k in np.arange(nvar):
        for j in np.arange(nvar):
            NI[j+1,k+1] = dt / b1**2. * np.sum(x[:,j] * x[:,k]) # inner elements of I
    for k in np.arange(nvar):
        NI[k+1,nvar+1] = 2. * dt / b1**3. * np.sum(R1 * x[:,k]) # inner elements of last column of I
    for j in np.arange(nvar+2):
        for k in np.arange(j):
            NI[j,k] = NI[k,j] # fill elements left of diagonal of I (symmetry)
    invNI = np.linalg.inv(NI) # covariance matrix (inverse of Fisher information matrix)
    var_a12 = invNI[2,2] # variance of a12
    sd_a12 = np.sqrt(var_a12) # standard deviation of a12
    error_T21_FI = (C12 / C11) * sd_a12
    
    # Check that T21b = a12 * C12 / C11 = T21 computed via eq. (10) of Liang (2014)
#    T21b = a12 * C12 / C11 # check that T21b = T21
    
    # Check tau21
#    Z21b = np.abs(T21) + np.abs(a11) + np.abs(Q1*dt/(2.*C11*N))
#    tau21b = 100. * T21 / Z21b
    
    return T21,tau21,error_T21,error_tau21,R,error_R,error_T21_FI