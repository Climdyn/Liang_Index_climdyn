 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Function to compute the transfer of information from variable xj to variable xi (T) and the corresponding normalization (tau)
Applied to multivariate time series
Based on Liang (2021, 'Normalized multivariate time series causality analysis and causal graph reconstruction', Entropy, doi: 10.3390/e23060679)

Also compute Pearson correlation coefficient R

Compute the error in T_ji, tau_ji and R_ji based on the normal bootstrap with replacement (resampling of variables)

Inputs: 
x: matrix of variables with dimension nvar x N (nvar is the number of variables and N is the length of the time series)
dt: time step (e.g. 1 month)
n_iter: number of bootstrap realizations for computing the error

Outputs:
T: absolute rate of information transfer (dimension nvar x nvar)
tau: relative rate of information transfer (dimension nvar x nvar)
R: Pearson correlation coefficient (dimension nvar x nvar)
error_T: error in T (dimension nvar x nvar)
error_tau: error in tau (dimension nvar x nvar)
error_R: error in R (dimension nvar x nvar)

Created: 20/04/2021
Last updated: 23/11/2021

@author: David Docquier
"""

import numpy as np
import numba # translates Python and numpy code into fast machine code
from numba import jit

@jit(nopython=True) # accelerator
def compute_liang_nvar(x,dt,n_iter):
    
    # Function to compute absolute transfer of information from xj to xi (T)
    def compute_liang_index(detC,Deltajk,Ckdi,Cij,Cii):
        T = (1. / detC) * np.sum(Deltajk * Ckdi) * (Cij / Cii) # absolute rate of information flowing from xj to xi (nats per unit time) (equation (14))
        return T
    
    # Function to compute relative transfer of information from xj to xi (tau)
    def compute_liang_index_norm(detC,Deltaik,Ckdi,T_all,Tii,gii,Cii,Tji):
        selfcontrib = (1. / detC) * np.sum(Deltaik * Ckdi) # self-contribution (equation (15))
        transfer = np.sum(np.abs(T_all)) - np.abs(Tii) # all other transfers contribution (equation (20))
        noise = 0.5 * gii / Cii # noise contribution
        Z = np.abs(selfcontrib) + transfer + np.abs(noise) # normalizer (equation (20))
        tau = 100. * Tji / Z # relative rate of information flowing from xj to xi (%) (equation (19))
        return tau
    
    # Dimensions
    nvar = x.shape[0] # number of variables
    N = x.shape[1] # length of the time series (number of observations)
    
    # Compute tendency dx
    k = 1 # k = 1 (or 2 for highly chaotic and densely sampled systems)
    dx = np.zeros((nvar,N)) # initialization of dx (to have the same number of time steps as x)
    for i in np.arange(nvar):
        dx[i,0:N-k] = (x[i,k:N] - x[i,0:N-k]) / (k * dt) # Euler forward finite difference of x (equation (7))
    
    # Compute covariances and matrix determinant
    C = np.cov(x) # covariance matrix
    dC = np.empty_like(C) * 0.
    for i in np.arange(nvar):
        for j in np.arange(nvar):
            dC[j,i] = (np.sum((x[j,:] - np.nanmean(x[j,:])) * (dx[i,:] - np.nanmean(dx[i,:])))) / (N - 1.) # covariance between x and dx
    detC = np.linalg.det(C) # matrix determinant
    
    # Compute cofactors
    Delta = np.linalg.inv(C).T * detC # cofactor matrix (https://en.wikipedia.org/wiki/Minor_(linear_algebra))
    
    # Compute absolute transfer of information (T) and correlation coefficient (R)
    T = np.zeros((nvar,nvar))
    R = np.zeros((nvar,nvar))
    for i in np.arange(nvar):
        for j in np.arange(nvar):
            T[j,i] = compute_liang_index(detC,Delta[j,:],dC[:,i],C[i,j],C[i,i]) # compute T (transfer of information from xj to xi) and create matrix
            R[j,i] = C[i,j] / np.sqrt(C[i,i] * C[j,j]) # compute correlation coefficient and create correlation matrix
    
    # Compute noise terms
    g = np.zeros(nvar)
    for i in np.arange(nvar):
        a1k = np.dot(np.linalg.inv(C),dC[:,i]) # compute a1k coefficients based on matrix-vector product (see beginning of page 4 in Liang (2014))
        f1 = np.nanmean(dx[i,:])
        for k in np.arange(nvar):
            f1 = f1 - a1k[k] * np.nanmean(x[k,:])
        R1 = dx[i,:] - f1
        for k in np.arange(nvar):
            R1 = R1 - a1k[k] * x[k,:]
        Q1 = np.sum(R1**2.)       
        g[i] = Q1 * dt / N # equation (10)
    
    # Compute relative transfer of information (tau)
    tau = np.zeros((nvar,nvar))
    for i in np.arange(nvar):
        for j in np.arange(nvar):
            tau[j,i] = compute_liang_index_norm(detC,Delta[i,:],dC[:,i],T[:,i],T[i,i],g[i],C[i,i],T[j,i]) # compute tau and create matrix
    
    # Initialize error in T, tau and R
    boot_T = np.zeros((n_iter,nvar,nvar))
    boot_tau = np.zeros((n_iter,nvar,nvar))
    boot_R = np.zeros((n_iter,nvar,nvar))
    
    # Compute error in T, tau and R using bootstrap with replacement
    for it in np.arange(n_iter): # loop over bootstrap realizations

        # Resample x and dx
        index = np.arange(N)
        boot_index = np.random.choice(index,N,replace=True) # resample index
        boot_x = np.zeros((nvar,N))
        boot_dx = np.zeros((nvar,N))
        for t in np.arange(N):
            boot_x[:,t] = x[:,boot_index[t]] # x corresponding to resampled index
            boot_dx[:,t] = dx[:,boot_index[t]] # dx corresponding to resampled index
        
        # Compute covariances and matrix determinant based on resampled variables
        boot_C = np.cov(boot_x)
        boot_dC = np.empty_like(boot_C) * 0.
        for i in np.arange(nvar):
            for j in np.arange(nvar):
                boot_dC[j,i] = (np.sum((boot_x[j,:] - np.nanmean(boot_x[j,:])) * (boot_dx[i,:] - np.nanmean(boot_dx[i,:])))) / (N - 1.)
        boot_detC = np.linalg.det(boot_C)
        
        # Compute cofactors based on resampled variables
        boot_Delta = np.linalg.inv(boot_C).T * boot_detC

        # Compute absolute transfer of information (T) and correlation coefficient (R) based on resampled variables
        for i in np.arange(nvar):
            for j in np.arange(nvar):
                boot_T[it,j,i] = compute_liang_index(boot_detC,boot_Delta[j,:],boot_dC[:,i],boot_C[i,j],boot_C[i,i])
                boot_R[it,j,i] = boot_C[i,j] / np.sqrt(boot_C[i,i] * boot_C[j,j])
        
        # Compute noise terms based on resampled variables
        boot_g = np.zeros(nvar)
        for i in np.arange(nvar):
            a1k = np.dot(np.linalg.inv(boot_C),boot_dC[:,i])
            f1 = np.nanmean(boot_dx[i,:])
            for k in np.arange(nvar):
                f1 = f1 - a1k[k] * np.nanmean(boot_x[k,:])
            R1 = boot_dx[i,:] - f1
            for k in np.arange(nvar):
                R1 = R1 - a1k[k] * boot_x[k,:]
            Q1 = np.sum(R1**2.)       
            boot_g[i] = Q1 * dt / N

        # Compute relative transfer of information (tau) based on resampled variables
        for i in np.arange(nvar):
            for j in np.arange(nvar):
                boot_tau[it,j,i] = compute_liang_index_norm(boot_detC,boot_Delta[i,:],boot_dC[:,i],boot_T[it,:,i],boot_T[it,i,i],boot_g[i],boot_C[i,i],boot_T[it,j,i])

    # Compute error in T, tau and R (standard deviation of boostraped values)
    error_T = np.nanstd(boot_T,axis=0)
    error_tau = np.nanstd(boot_tau,axis=0)
    error_R = np.nanstd(boot_R,axis=0)
    
    # Return result of function
    return T,tau,R,error_T,error_tau,error_R