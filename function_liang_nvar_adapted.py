#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Function to compute the Liang-Kleeman information flow (LKIF) from variable xj to variable xi (T) and the corresponding normalization (tau)
Applied to multivariate time series
Based on Liang (2021, 'Normalized multivariate time series causality analysis and causal graph reconstruction', Entropy, https://doi.org/10.3390/e23060679)
Code partly based on Liang's Matlab code (multi_causality_est.m)

Compute the error in T_ji and tau_ji:
    - either on normal bootstrap with replacement (resampling of variables) (Docquier et al., 2024; https://doi.org/10.5194/npg-31-115-2024)
    - or on the Fisher Information Matrix (FIM; Liang, 2021)

Inputs: 
x: matrix of variables with dimension nvar x N (nvar is the number of variables and N is the length of the time series)
dx: dx/dt (tendency)
dt: time step (e.g. 1 month or 1 day)
error_computation: 1 if boostrap resampling; 2 if FIM
n_iter: number of bootstrap realizations for computing the error (only used if error_computation = 1)

Outputs:
T: absolute rate of information transfer (dimension nvar x nvar)
tau: relative rate of information transfer (dimension nvar x nvar)
error_T: error in T (dimension nvar x nvar)
error_tau: error in tau (dimension nvar x nvar)

Last updated: 12/02/2026

@author: David Docquier
"""

import numpy as np

def compute_liang_nvar(x,dx,dt,error_computation,n_iter):
    
    # Dimensions
    nvar = x.shape[0] # number of variables
    N = x.shape[1] # length of the time series (number of observations)
    
    # Compute covariances and matrix determinant
    C = np.cov(x) # covariance matrix
    dC = np.empty_like(C) * 0. # covariance between x and dx/dt
    for i in np.arange(nvar):
        for j in np.arange(nvar):
            dC[j,i] = (np.sum((x[j,:] - np.nanmean(x[j,:])) * (dx[i,:] - np.nanmean(dx[i,:])))) / (N - 1.) # covariance between x and dx/dt

    # Compute absolute (T) and relative (tau) rates of information transfer as well as noise term (g)
    T = np.zeros((nvar,nvar))
    tau = np.zeros((nvar,nvar))
    g = np.zeros(nvar)
    R1 = np.zeros((nvar,N))
    for i in np.arange(nvar):
        a1k = np.dot(np.linalg.inv(C),dC[:,i]) # estimate a1k coefficients (maximum-likelihood estimator) based on matrix-vector product (see beginning of page 4 in Liang (2014) and Liang's Matlab code a1n)
        
        # Absolute rate of information transfer (T)
        for j in np.arange(nvar):
            T[j,i] = a1k[j] * C[i,j] / C[i,i] # asolute rate of information transfer T(j->i) - eq. (4) of Liang (2021) - same result as using eq. (14) of Liang (2021)
        
        # Noise term (g)
        f1 = np.nanmean(dx[i,:])
        for k in np.arange(nvar):
            f1 = f1 - a1k[k] * np.nanmean(x[k,:]) # eq. (11) of Liang (2021)
        R1[i,:] = dx[i,:] - f1
        for k in np.arange(nvar):
            R1[i,:] = R1[i,:] - a1k[k] * x[k,:] # beginning of page 6 in Liang (2021)
        Q1 = np.sum(R1[i,:]**2.) # end of page 6 in Liang (2021)
        g[i] = Q1 * dt / N # noise term - eq. (10) of Liang [2021]
    
        # Normalization - Relative rate of information transfer (tau)
        sum_transfer = np.sum(np.abs(T[:,i])) # sum of all transfers from the drivers to the target variable x_i
        noise = 0.5 * g[i] / C[i,i] # noise contribution - end of page 8 in Liang (2021)
        Z = sum_transfer + np.abs(noise) # normalizer (eq. (20) of Liang [2021]) - the self-influence (dH1*/dt) is included in sum_transfer
        tau[:,i] = 100. * T[:,i] / Z # relative rate of information transfer tau(j->i) - eq. (19) of Liang [2021]
    
    # Compute error in T and tau
    error_T = np.zeros((nvar,nvar))
    error_tau = np.zeros((nvar,nvar))
    
    if error_computation == 1: # using bootstrap with replacement (as in Docquier et al. [2024])
    
        # Initialize bootstrapped T and tau
        boot_T = np.zeros((n_iter,nvar,nvar))
        boot_tau = np.zeros((n_iter,nvar,nvar))
        
        # Loop over bootstrap realizations
        for it in np.arange(n_iter):
    
            # Resample x and dx/dt
            index = np.arange(N)
            boot_index = np.random.choice(index,N,replace=True) # resample index
            boot_x = np.zeros((nvar,N))
            boot_dx = np.zeros((nvar,N))
            for t in np.arange(N):
                boot_x[:,t] = x[:,boot_index[t]] # x corresponding to resampled index
                boot_dx[:,t] = dx[:,boot_index[t]] # dx/dt corresponding to resampled index
            
            # Compute covariances and matrix determinant based on resampled variables
            boot_C = np.cov(boot_x)
            boot_dC = np.empty_like(boot_C) * 0.
            for i in np.arange(nvar):
                for j in np.arange(nvar):
                    boot_dC[j,i] = (np.sum((boot_x[j,:] - np.nanmean(boot_x[j,:])) * (boot_dx[i,:] - np.nanmean(boot_dx[i,:])))) / (N - 1.)
            
            # Compute absolute / relative rates of information transfer and noise terms based on resampled variables
            g = np.zeros(nvar)
            R1 = np.zeros((nvar,N))
            for i in np.arange(nvar):
                a1k = np.dot(np.linalg.inv(boot_C),boot_dC[:,i])
                
                for j in np.arange(nvar):
                    boot_T[it,j,i] = a1k[j] * boot_C[i,j] / boot_C[i,i]
                
                f1 = np.nanmean(boot_dx[i,:])
                for k in np.arange(nvar):
                    f1 = f1 - a1k[k] * np.nanmean(boot_x[k,:])
                R1[i,:] = boot_dx[i,:] - f1
                for k in np.arange(nvar):
                    R1[i,:] = R1[i,:] - a1k[k] * boot_x[k,:]
                Q1 = np.sum(R1[i,:]**2.)
                g[i] = Q1 * dt / N
                
                sum_transfer = np.sum(np.abs(boot_T[it,:,i]))
                noise = 0.5 * g[i] / boot_C[i,i]
                Z = sum_transfer + np.abs(noise)
                boot_tau[it,:,i] = 100. * boot_T[it,:,i] / Z
    
        # Compute error in T and tau (standard deviation of boostraped values)
        error_T = np.nanstd(boot_T,axis=0)
        error_tau = np.nanstd(boot_tau,axis=0)
    
    elif error_computation == 2: # using Fisher Information Matrix (FIM; as in Liang [2021])
        
        # Compute error in T based on the Fisher information matrix I (error in tau is not computed)
        b1 = np.sqrt(g)
        var_a1k = np.zeros((nvar,nvar))
        for i in np.arange(nvar):
            NI = np.zeros((nvar+2,nvar+2))
            NI[0,0] = N * dt / b1[i]**2. # 1st element of 1st row of I
            NI[nvar+1,nvar+1] = 3. * dt / b1[i]**4. * np.sum(R1[i,:]**2.) - N / b1[i]**2. # last element of last row of I
            for k in np.arange(nvar):
                NI[0,k+1] = dt / b1[i]**2. * np.sum(x[k,:]) # 1st row of I
            NI[0,nvar+1] = 2. * dt / b1[i]**3. * np.sum(R1[i,:]) # last element of 1st row of I
            for k in np.arange(nvar):
                for j in np.arange(nvar):
                    NI[j+1,k+1] = dt / b1[i]**2. * np.sum(x[j,:] * x[k,:]) # inner elements of I
            for k in np.arange(nvar):
                NI[k+1,nvar+1] = 2. * dt / b1[i]**3. * np.sum(R1[i,:] * x[k,:]) # inner elements of last column of I
            for j in np.arange(nvar+2):
                for k in np.arange(j):
                    NI[j,k] = NI[k,j] # fill elements left of diagonal of I (symmetry)
            invNI = np.linalg.inv(NI) # covariance matrix of theta (inverse of Fisher information matrix)
            for k in np.arange(nvar):
                var_a1k[i,k] = (C[i,k] / C[i,i])**2. * invNI[k+1,k+1] # variance in a1k coefficients
                error_T[k,i] = np.sqrt(var_a1k[i,k]) # error in T - end of page 7 in Liang (2021) 
    
    # Return result of function
    return T,tau,error_T,error_tau