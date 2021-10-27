'''
==============================================================================
                        Concrete Classes
        Pre-scribed templates that are ready for pricing and tuning!
'''

import numpy as np
from .._stochasticprocess import stochastic_process
from .._dimexpansions import broadcast, value

__all__ = ["white_noise","GBM","CIR"]

_rng = np.random.default_rng()

class white_noise ( stochastic_process ):
    
    '''
    Gaussian white noise
    https://en.wikipedia.org/wiki/White_noise
    '''
    
    def __init__(self,
                 n_assets : int = 1,
                 corr_mat : list = None
                 ):

        super().__init__()
        
        # create correlation matrix if none given
        if corr_mat is None:
            corr_mat = np.diag(np.ones(n_assets))
        
        self.__parameter__("n_assets",
                           expected_shape = (),
                           init_value = n_assets,
                           broadcastable = False,
                           dim_expansion_by = value(larger_than=1),
                           expect_simdim = False,
                           expect_timedim = False
                           )
        
        self.__parameter__("corr_mat",
                           expected_shape = (n_assets,n_assets),
                           init_value = corr_mat,
                           broadcastable = False,
                           dim_expansion_by = None,
                           expect_simdim = False,
                           expect_timedim = False )
        
        full_output_structure = ("n_walks","steps","n_assets")
        self.__compile__( full_output_structure )
    
    def __str__(self):
        dims = self.parameters["n_assets"].value
        corr_mat = self.parameters.get("corr_mat")
        corred = "" if np.count_nonzero(corr_mat) - dims else "un"
        return f"{corred}correlated {dims}-d White Noise"
        
    def __walk__(self):
        
        # unpack
        steps    = self.steps 
        n_walks  = self.n_walks
        n_assets = self.n_assets
        corr_mat = self.corr_mat
        
        # draw
        if n_assets == 1:
            noise = _rng.normal( 0, 1, [ n_walks, steps ] )
        else:
            means = np.zeros( n_assets )
            noise = _rng.multivariate_normal(means, corr_mat, [n_walks, steps])        
        
        return noise
    
_white_noise = white_noise()

class poisson_process ( stochastic_process ):
    
    def __init__(self,
                 lamb : float,
                 ):
                
        '''
        Poisson point process
        https://en.wikipedia.org/wiki/Poisson_point_process#Simulation
        
        params: 
            s0 = Spot level
            v = Contant variance
            kappa - Mean reversion rate
            theta - Long term level
        '''

        super().__init__()
        
        self.__parameter__("lamb", (),
                           init_value = lamb,
                           broadcastable = True, 
                           dim_expansion_by = broadcast)
                
        full_output_structure = ("n_walks",
                                 "steps",
                                 "lamb")
        
        self.__compile__( full_output_structure )
    
    def __str__(self):
        return "Poisson Point Process"
    
    def __walk__(self):
        
        # unpack
        dt    = self.dt
        lamb = self.lamb
        
        P  = _rng.poisson(lam = lamb * dt, size = lamb.shape )
        return P

class GBM ( stochastic_process ):

    '''
    Geometric Brownian Motion
    https://en.wikipedia.org/wiki/Geometric_Brownian_motion
    
    required params: 
        s0 = Spot level at t=0
        mu = (annualized) drift
        sigma = (annualized) stddev
        
    optionals:
        noise = process driving the randomness of increments
                defaults to white noise
    '''
    
    def __init__(self,
                 s0 : float or 'stochastic_process',
                 mu : float or 'stochastic_process',
                 sigma : float or 'stochastic_process',
                 noise : 'stochastic_process' = _white_noise,
                 ):
        
        super().__init__()
        
        self.__parameter__("s0", (), 
                           init_value = s0, 
                           broadcastable = True, 
                           dim_expansion_by = broadcast,
                           expect_simdim=None,
                           expect_timedim=False)
        
        self.__parameter__("mu", (),
                           init_value = mu,
                           broadcastable = True, 
                           dim_expansion_by = broadcast)
        
        self.__parameter__("sigma", (),
                           init_value = sigma, 
                           broadcastable = True, 
                           dim_expansion_by = broadcast )
        
        self.__parameter__("noise", (),
                           init_value = noise, 
                           broadcastable = True, 
                           dim_expansion_by = broadcast,
                           expect_simdim = True,
                           expect_timedim = True,
                           )
        
        full_output_structure = ("n_walks","steps","noise","s0","mu","sigma")
        self.__compile__( full_output_structure )
    
    def __str__(self):
        return "Geometric Brownian Motion"
    
    def __walk__(self):
        
        # unpack
        dt    = self.dt
        noise = self.noise
        s0    = self.s0
        mu    = self.mu
        sigma = self.sigma
               
        drift = ( mu - 0.5 * sigma**2 ) * dt
        diffusion = noise * sigma * np.sqrt(dt)
        increments = np.exp( drift + diffusion )
        
        walks = s0 * np.cumprod( increments, axis = 1 )
        
        return walks
    
class OU ( stochastic_process ):
    
    def __init__(self,
                 s0 : float or "stochastic_process",
                 sigma : float or "stochastic_process",
                 kappa : float or "stochastic_process",
                 theta : float or "stochastic_process",
                 noise : 'stochastic_process' = _white_noise,                 
                 ):
                
        '''
        Ohnstein Uhlenbeck process
        https://en.wikipedia.org/wiki/Cox%E2%80%93Ingersoll%E2%80%93Ross_model
        
        params: 
            s0 = Spot level
            v = Contant variance
            kappa - Mean reversion rate
            theta - Long term level
        '''
    
        super().__init__()
        
        self.__parameter__("s0", (), 
                           init_value = s0,
                           broadcastable = True, 
                           dim_expansion_by = broadcast,
                           expect_sim_dims = False )
        
        self.__parameter__("sigma", (),
                           init_value = sigma,
                           broadcastable = True, 
                           dim_expansion_by = broadcast)
        
        self.__parameter__("kappa", (),
                           init_value = kappa, 
                           broadcastable = True,
                           dim_expansion_by = broadcast )
        
        self.__parameter__("theta", (),
                           init_value = theta, 
                           broadcastable = True,
                           dim_expansion_by = broadcast )
        
        self.__parameter__("noise", (),
                           init_value = noise, 
                           broadcastable = True, 
                           dim_expansion_by = broadcast,
                           expect_sim_dims = True 
                           )
        
        full_output_structure = ("n_walks",
                                 "steps",
                                 "noise",
                                 "s0",
                                 "sigma",
                                 "kappa",
                                 "theta")
        
        self.__compile__( full_output_structure )
    
    def __str__(self):
        return "Ohnstein Uhlenbeck Process"
    
    def __walk__(self):
        
        # unpack
        dt    = self.dt
        steps = self.steps
        noise = self.noise
        s0    = self.s0
        sigma = self.sigma
        v     = sigma**2
        kappa = self.kappa
        theta = self.theta
        
        walks = []
        s_t = s0[:,0,None]
        for i in range(steps):
            diffusion = np.sqrt( v[:,i,None] * dt ) * noise[:,i,None]
            meanrev = kappa[:,i,None] * ( theta[:,i,None] - s_t ) * dt            
            s_t = s_t + meanrev + diffusion
            walks.append(s_t)
        walks = np.concatenate( walks, 1 )
        
        return walks

class CIR ( stochastic_process ):
    
    def __init__(self,
                 s0 : float or "stochastic_process",
                 sigma : float or "stochastic_process",
                 kappa : float or "stochastic_process",
                 theta : float or "stochastic_process",
                 noise : 'stochastic_process' = _white_noise,                 
                 ):
                
        '''
        Cox–Ingersoll–Ross mean reverting process
        https://en.wikipedia.org/wiki/Cox%E2%80%93Ingersoll%E2%80%93Ross_model
        
        params: 
            s0 = Spot level
            v = Contant variance
            kappa - Mean reversion rate
            theta - Long term level
        '''
    
        super().__init__()
        
        self.__parameter__("s0", (), 
                           init_value = s0,
                           broadcastable = True, 
                           dim_expansion_by = broadcast,
                           expect_sim_dims = False )
        
        self.__parameter__("sigma", (),
                           init_value = sigma,
                           broadcastable = True, 
                           dim_expansion_by = broadcast)
        
        self.__parameter__("kappa", (),
                           init_value = kappa, 
                           broadcastable = True,
                           dim_expansion_by = broadcast )
        
        self.__parameter__("theta", (),
                           init_value = theta, 
                           broadcastable = True,
                           dim_expansion_by = broadcast )
        
        self.__parameter__("noise", (),
                           init_value = noise, 
                           broadcastable = True, 
                           dim_expansion_by = broadcast,
                           expect_sim_dims = True 
                           )
        
        full_output_structure = ("n_walks",
                                 "steps",
                                 "noise",
                                 "s0",
                                 "sigma",
                                 "kappa",
                                 "theta")
        
        self.__compile__( full_output_structure )
    
    def __str__(self):
        return "Cox–Ingersoll–Ross Process"
    
    def __walk__(self):
        
        # unpack
        dt    = self.dt
        steps = self.steps
        noise = self.noise
        s0    = self.s0
        sigma = self.sigma
        v     = sigma**2
        kappa = self.kappa
        theta = self.theta
        
        walks = []
        s_t = s0[:,0,None]
        for i in range(steps):
            diffusion = np.sqrt( v[:,i,None] * dt ) * noise[:,i,None]
            meanrev = kappa[:,i,None] * ( theta[:,i,None] - s_t ) * dt            
            s_t = abs( s_t + meanrev + diffusion )
            walks.append(s_t)
        walks = np.concatenate( walks, 1 )
        
        return walks
    
class Bates ( stochastic_process ):

    '''
    Stochastic volatility jump model 
    https://en.wikipedia.org/wiki/Stochastic_volatility_jump
    
    params: 
        s0 = Spot price
        sigma0 = Spot Volatility
        mu - Drift
        rho - Correlation between price and volatility
        kappa - Mean reversion rate
        theta - Long term variance
        xi - vol of vol (variance)
        lamb - jump frequency
        k - Jump intensity
    '''
    
    def __init__(self,
                 s0 : float or 'stochastic_process',
                 sigma0 : float or 'stochastic_process',
                 mu : float or 'stochastic_process',
                 sigma : float or 'stochastic_process',
                 noise : 'stochastic_process' = _white_noise,
                 ):
        
        super().__init__()
        
        self.__parameter__("s0", (), 
                           init_value = s0, 
                           broadcastable = True, 
                           dim_expansion_by = broadcast,
                           expect_simdim=None,
                           expect_timedim=False)
        
        self.__parameter__("mu", (),
                           init_value = mu,
                           broadcastable = True, 
                           dim_expansion_by = broadcast)
        
        self.__parameter__("sigma", (),
                           init_value = sigma, 
                           broadcastable = True, 
                           dim_expansion_by = broadcast )
        
        self.__parameter__("noise", (),
                           init_value = noise, 
                           broadcastable = True, 
                           dim_expansion_by = broadcast,
                           expect_simdim = True,
                           expect_timedim = True,
                           )
        
        full_output_structure = ("n_walks","steps","noise","s0","mu","sigma")
        self.__compile__( full_output_structure )
    
    def __str__(self):
        return "Stochastic volatility jump model"
    
    def __walk__(self):
        
        # unpack
        dt    = self.dt
        noise = self.noise
        s0    = self.s0
        mu    = self.mu
        sigma = self.sigma
               
        drift = ( mu - 0.5 * sigma**2 ) * dt
        diffusion = noise * sigma * np.sqrt(dt)
        increments = np.exp( drift + diffusion )
        
        walks = s0 * np.cumprod( increments, axis = 1 )
        
        return walks
    