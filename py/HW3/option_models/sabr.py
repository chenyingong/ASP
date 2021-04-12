    # -*- coding: utf-8 -*-
"""
Created on Tue Oct 10

@author: jaehyuk
"""

import numpy as np
import scipy.stats as ss
import scipy.optimize as sopt
import scipy.integrate as spint
from . import normal
from . import bsm
import pyfeng as pf

'''
MC model class for Beta=1
'''
class ModelBsmMC:
    beta = 1.0   # fixed (not used)
    vov, rho = 0.0, 0.0
    sigma, intr, divr = None, None, None
    bsm_model = None
    '''
    You may define more members for MC: time step, etc
    '''
    
    def __init__(self, sigma, vov=0, rho=0.0, beta=1.0, intr=0, divr=0, time_steps=1_000, n_samples=10_000):
        self.sigma = sigma
        self.vov = vov
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.time_steps = time_steps
        self.n_samples = n_samples
        self.bsm_model = pf.Bsm(sigma, intr=intr, divr=divr)
        
    def bsm_vol(self, strike, spot, texp=None, sigma=None):
        '''
        From the price from self.price() compute the implied vol
        this is the opposite of bsm_vol in ModelHagan class
        use bsm_model
        '''
        if sigma is None:
            sigma = self.sigma
        price = self.price(strike, spot, texp, sigma)
        vol = self.bsm_model.impvol(price, strike, spot, texp)
        return vol
    
    def price(self, strike, spot, texp=None, sigma=None, cp=1, seed=None):
        '''
        Your MC routine goes here
        Generate paths for vol and price first. Then get prices (vector) for all strikes
        You may fix the random number seed
        '''
        if seed is not None:
            np.random.seed(seed)

        if sigma is None:
            sigma = self.sigma

        znorm_m = np.random.normal(size=(self.n_samples, ))
        X1 = np.random.normal(loc=0., scale=1., size=(self.n_samples,))

        # generate path for vol
        n_intervals = np.linspace(0, texp, self.time_steps)
        vol_path = sigma * np.exp(self.vov * znorm_m - 1/2 * (self.vov**2) * n_intervals[:, None])

        div_fac = np.exp(-texp * self.divr)
        disc_fac = np.exp(-texp * self.intr)
        forward = spot / disc_fac * div_fac

        # compute V_T by Simpon's method
        # V_T = (np.sum(2*vol_path**2, axis=0) - vol_path[-1] - vol_path[0]) * (texp / (self.time_steps - 1)) / 2
        V_T = spint.simps(vol_path**2, dx=texp / (self.time_steps - 1), axis=0)
        # generate path for price
        S_T = np.exp(np.log(forward) + self.rho / self.vov * (vol_path[-1] - sigma) +
                     np.sqrt(1 - self.rho**2) * X1 * np.sqrt(V_T) - 1 / 2 * V_T)
        price = np.mean(np.fmax(cp*(S_T - strike[:, None]), 0), axis=1)
        return disc_fac * price

'''
MC model class for Beta=0
'''
class ModelNormalMC:
    beta = 0.0   # fixed (not used)
    vov, rho = 0.0, 0.0
    sigma, intr, divr = None, None, None
    normal_model = None
    
    def __init__(self, sigma, vov=0, rho=0.0, beta=0.0, intr=0, divr=0, time_steps=1_000, n_samples=10_000):
        self.sigma = sigma
        self.vov = vov
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.time_steps = time_steps
        self.n_samples = n_samples
        self.normal_model = pf.Norm(sigma, intr=intr, divr=divr)
        
    def norm_vol(self, strike, spot, texp=None, sigma=None):
        '''
        From the price from self.price() compute the implied vol
        this is the opposite of normal_vol in ModelNormalHagan class
        use normal_model 
        '''
        if sigma is None:
            sigma = self.sigma
        price = self.price(strike, spot, texp, sigma)
        vol = self.normal_model.impvol(price, strike, spot, texp)
        return vol
        
    def price(self, strike, spot, texp=None, sigma=None, cp=1, seed=None):
        '''
        Your MC routine goes here
        Generate paths for vol and price first. Then get prices (vector) for all strikes
        You may fix the random number seed
        '''
        if seed is not None:
            np.random.seed(seed)

        if sigma is None:
            sigma = self.sigma

        znorm_m = np.random.normal(size=(self.n_samples, ))
        X1 = np.random.normal(loc=0., scale=1., size=(self.n_samples,))

        # generate path for vol
        n_intervals = np.linspace(0, texp, self.time_steps)
        vol_path = sigma * np.exp(self.vov * znorm_m - 1/2 * (self.vov**2) * n_intervals[:, None])

        div_fac = np.exp(-texp * self.divr)
        disc_fac = np.exp(-texp * self.intr)
        forward = spot / disc_fac * div_fac

        # compute V_T by Simpon's method
        # V_T = (np.sum(2*vol_path**2, axis=0) - vol_path[-1] - vol_path[0]) * (texp / (self.time_steps - 1)) / 2
        V_T = spint.simps(vol_path**2, dx=texp / (self.time_steps - 1), axis=0)
        # generate path for price
        S_T = forward + self.rho / self.vov * (vol_path[-1] - sigma) + np.sqrt(1 - self.rho**2) * X1 * np.sqrt(V_T)
        price = np.mean(np.fmax(cp*(S_T - strike[:, None]), 0), axis=1)
        return disc_fac * price

'''
Conditional MC model class for Beta=1
'''
class ModelBsmCondMC:
    beta = 1.0   # fixed (not used)
    vov, rho = 0.0, 0.0
    sigma, intr, divr = None, None, None
    bsm_model = None
    '''
    You may define more members for MC: time step, etc
    '''
    
    def __init__(self, sigma, vov=0, rho=0.0, beta=1.0, intr=0, divr=0, time_steps=1_000, n_samples=10_000):
        self.sigma = sigma
        self.vov = vov
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.time_steps = time_steps
        self.n_samples = n_samples
        self.bsm_model = pf.Bsm(sigma, intr=intr, divr=divr)
        
    def bsm_vol(self, strike, spot, texp=None):
        ''''
        From the price from self.price() compute the implied vol
        this is the opposite of bsm_vol in ModelHagan class
        use bsm_model
        should be same as bsm_vol method in ModelBsmMC (just copy & paste)
        '''
        price = self.price(strike, spot, texp)
        vol = self.bsm_model.impvol(price, strike, spot, texp)
        return vol
    
    def price(self, strike, spot, texp=None, cp=1, seed=None):
        '''
        Your MC routine goes here
        Generate paths for vol only. Then compute integrated variance and BSM price.
        Then get prices (vector) for all strikes
        You may fix the random number seed
        '''
        if seed is not None:
            np.random.seed(seed)

        # generate path for vol
        znorm_m = np.random.normal(size=(self.n_samples, ))
        n_intervals = np.linspace(0, texp, self.time_steps)
        vol_path = self.sigma * np.exp(self.vov * znorm_m - 1/2 * (self.vov**2) * n_intervals[:, None])

        # compute integrated variance
        assert self.time_steps % 2 == 0, "N should be even"
        temp = 0
        coefficients = [1, 4, 2]
        for i in range(self.time_steps):
            temp += (coefficients[i % 3] * vol_path[i]**2)
        I_T = 1 / (3*self.time_steps*self.sigma**2) * temp

        spot *= np.exp(self.rho / self.vov * (vol_path[-1] - self.sigma) -
                       1 / 2 * self.rho**2 * self.sigma**2 * texp * I_T)
        vol = self.sigma * np.sqrt((1 - self.rho**2) * I_T)

        price = bsm.price(strike, spot, texp, vol, self.intr, self.divr, cp)
        price = np.mean(price, axis=0)

        return price

'''
Conditional MC model class for Beta=0
'''
class ModelNormalCondMC:
    beta = 0.0   # fixed (not used)
    vov, rho = 0.0, 0.0
    sigma, intr, divr = None, None, None
    normal_model = None
    
    def __init__(self, sigma, vov=0, rho=0.0, beta=0.0, intr=0, divr=0, time_steps=1_000, n_samples=10_000):
        self.sigma = sigma
        self.vov = vov
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.time_steps = time_steps
        self.n_samples = n_samples
        self.normal_model = pf.Norm(sigma, intr=intr, divr=divr)
        
    def norm_vol(self, strike, spot, texp=None):
        ''''
        From the price from self.price() compute the implied vol
        this is the opposite of normal_vol in ModelNormalHagan class
        use normal_model
        should be same as norm_vol method in ModelNormalMC (just copy & paste)
        '''
        price = self.price(strike, spot, texp)
        vol = self.normal_model.impvol(price, strike, spot, texp)
        return vol
        
    def price(self, strike, spot, texp=None, cp=1, seed=None):
        '''
        Your MC routine goes here
        Generate paths for vol only. Then compute integrated variance and normal price.
        You may fix the random number seed
        '''
        if seed is not None:
            np.random.seed(seed)

        # generate path for vol
        znorm_m = np.random.normal(size=(self.n_samples, ))
        n_intervals = np.linspace(0, texp, self.time_steps)
        vol_path = self.sigma * np.exp(self.vov * znorm_m - 1/2 * (self.vov**2) * n_intervals[:, None])

        # compute integrated variance
        I_T = 1 / (2*self.time_steps*self.sigma**2) * (np.sum(2*vol_path**2, axis=0) - vol_path[0] - vol_path[-1])

        spot = spot + self.rho / self.vov * (vol_path[-1] - self.sigma)
        vol = self.sigma * np.sqrt((1 - self.rho**2) * I_T)

        price = normal.price(strike, spot, texp, vol, self.intr, self.divr, cp)
        price = np.mean(price, axis=0)

        return price
