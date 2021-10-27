"""
  -- The great Monty for Python -- 
A library for convenient and flexible 
Monte Carlo simulation and estimation 
of stuff that derives thereof

Version 3.0
Created on Tue Feb  2 23:10:51 2021
@author: arthur boeoeoeoeoek

General principles:
    
    every degree of freedom has an expected shape but can 
    be with an extra dimension, which results in a new 
    dimension in the eventual output

    because every function requires data, every "operation"
    will take as a first input the data on which it is 
    performed. This is either passed as a tensor itself
    or as a distribution from which the data is then created

"""

#TODO error messages

# general
import numpy as np
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass

# parallelization
from multiprocessing import Pool

__sim_dims__ = { "n_walks", "steps" }

class parameter:
    '''
    Container for parameters that describe the stochastic process        
    '''
    def __init__(self,
                 name : str,
                 value : np.ndarray or list or 'stochastic_process',                 
                 ):
                
        self.name = name
        self.value = value
        self.fill_dims = ()
        self.outer_process = None
               
        # derived information from passed args
        self.is_stochastic = isinstance(value,stochastic_process)
                        
        shape = value.output_shape if self.is_stochastic else np.shape(value)
        self.shape = shape
        
    def __call__(self,**kwargs):
        value = self.value
        if self.is_stochastic: value = value.price(**kwargs)
        value = np.expand_dims(value, self.fill_dims)
        if self.fill_n_walks: value = np.repeat( value, kwargs["n_walks"], 0 ) # TODO
        if self.fill_steps: value = np.repeat( value, kwargs["steps"], 1 )     # TODO    
        return value
                  
    def __str__(self):
        return f"{self.name}={self.value}"

    def __repr__(self):
        if self.outer_process is None:
            return f"Unbound parameter {str(self)}"
        else:
            return f"Parameter {str(self)} for {self.outer_process}"

class stochastic_process(metaclass=ABCMeta):
    
    '''
    A meta class for how a stochastic process functions
    To subclass a stochastic process, the __init__ and 
    __walk__ should be defined with parameters and the 
    function should be defined.
    '''
    
    @dataclass(frozen=True)
    class __param_signature__:
        expected_shape : dict 
        broadcastable : bool
        dim_expansion_by : callable
        expect_simdim : bool
        expect_timedim : bool

    def __init__(self):
        super().__setattr__( "parameters", dict() )
        super().__setattr__( "signature", dict() )
     
    def __repr__( self ):
        param_string = ", ".join(map(str,self.parameters.values()))
        return f"{str(self)} ({param_string})"
    
    def __parameter__(self,                      
                      name : str,
                      expected_shape : tuple,
                      broadcastable : bool = False,
                      init_value : list or 'stochastic_process' = None,
                      expect_simdim : bool = None,
                      expect_timedim : bool = None,
                      dim_expansion_by : callable = None,
                      *args,**kwargs ):
        
        if not isinstance(expected_shape,(tuple,list,np.ndarray)):
            t = str(type(expected_shape))
            raise ValueError("Unknown type %s received for expected_shape" % t)
        
        self.signature[name] = self.__param_signature__(
            expected_shape = expected_shape,
            broadcastable = broadcastable,
            expect_simdim = expect_simdim,
            expect_timedim = expect_timedim,
            dim_expansion_by = dim_expansion_by,
        )
        
        if init_value is not None: 
            self.__setattr__(name, init_value)
                
    def __setattr__(self, name, value):
        
        # assert belonging
        try: param_sig = self.signature[ name ]
        except: raise ValueError # doesnt belong in process
        
        # unpack signature
        E_shape = param_sig.expected_shape
        E_ndim = len(E_shape)
        
        # bind parameter to process
        if not isinstance(value,parameter): value = parameter( name, value )
        value.outer_process = self
        G_shape = list( value.shape ) # given shape
            
        # assert shape
        shapes = zip( G_shape, E_shape )
        if any(G not in [E,None] for G,E in shapes): raise "shape error"
            
        # determine excess dims
        allowed_excess_dims = {0} # unless broadcast is triggered
        if param_sig.broadcastable: allowed_excess_dims.add(1)
        
        # handle simdims
        dimflags = param_sig.expect_simdim, param_sig.expect_timedim
        for dimname, dimflag in zip( ["n_walks","steps"] , dimflags ):            
            hasdim = dimname in G_shape # is the sim dim in the shape
            if hasdim: G_shape.remove(dimname) # drop for excess dim count
            if type(dimflag) is bool and hasdim != dimflag: raise ValueError
            repeat_axis = dimflag in [None,True] and not hasdim
            setattr(value,"fill_%s" % dimname, repeat_axis ) # to be repeated
                
        # check broadcast trigger 
        excess_dims = len(G_shape) - E_ndim
        if excess_dims not in allowed_excess_dims: raise ValueError        
        value.broadcasted = excess_dims == 1
        
        # output
        exp_dims_by = param_sig.dim_expansion_by
        value.expanded = () if exp_dims_by is None else exp_dims_by(value)
        
        # recompile if the parameters is updated
        recompile = name in self.parameters
        self.parameters[name] = value # GREAT JOB MR. PARAM. YOU MADE IT!
        if recompile: self.__compile__(self.full_output_structure)
                                
    def __compile__(self,full_output_structure,*args,**kwargs):
        
        # set final values 
        sign = self.signature
        bcable_params = sorted([p for p,s in sign.items() if s.broadcastable],
                               key = lambda p: full_output_structure.index(p))
        super().__setattr__("broadcastable_parameters", tuple(bcable_params))
        super().__setattr__("full_output_structure", full_output_structure)
        
        # unpack own output shape
        out_dims = self.output_dims
        out_shape = self.output_shape
        out_ndim = len( out_shape )
        
        # compile parameters
        for k,param in self.parameters.items():
            
            param_shape = param.shape
            param_signature = sign[k]
            if param_signature.broadcastable:
                
                # mapping the parameter dims to the stoch process output shape
                p_axes = [out_dims.index(dim) if dim in out_dims 
                          else out_dims.index(k) for dim in param_shape]
                
                # the axes that need to be filled
                missing = lambda i: i not in p_axes
                param.fill_dims = tuple(filter(missing, range(out_ndim)))
    
    @abstractmethod
    def __walk__(self,*args,**kwargs):
        '''
        This function defines how the subclass generates its walks
        Must be implemented in the subclass
        '''
        pass
    
    def price(self,
              n_walks : int,
              T : float,
              dt : float = None,
              batch_size : int = 1000,
              verbose : bool = True,
              *args, **kwargs ):
        
        #TODO adjust small error from step counts in case of T%dt!=0
        dt = T if dt is None else dt
        sim_info = {
            "T" : T, 
            "dt" : dt,
            "steps" : int(T/dt), # easy-to-access steps count
            "batch_size" : None, # no batching in inner calls
            "verbose" : False    # no pbars in inner calls
        }
        
        # determine batch sizes
        if batch_size is None:
            batch_sizes = [n_walks]
        else:
            full_batches = n_walks // batch_size
            modulo = n_walks%batch_size
            batch_sizes = [batch_size] * full_batches + [modulo]
        
        walk_batches = []
        for batch in batch_sizes:
            
            # update the batch_size
            sim_info["n_walks"] = batch
            
            # draw the broadcasted parameters
            sim_params = {n:p(**sim_info) for n,p in self.parameters.items()}
            
            # inject into namespace (temporary)
            self.__dict__.update( **sim_params, **sim_info )
            
            walk_batches.append( self.__walk__() )
                    
        walk_batches = np.concatenate(walk_batches,0)
        
        # clear namespace
        [ self.__dict__.pop(p) for p in [*sim_params, *sim_info] ]
        
        return walk_batches
        
    @property
    def expanded_dims(self):
        exp_dims = sorted((n for n,p in self.parameters.items() if p.expanded),
                          key = lambda p: self.full_output_structure.index(p))
        return tuple( exp_dims )
            
    @property
    def output_dims(self):
        in_output = lambda dim: dim in tuple(__sim_dims__) + self.expanded_dims
        return tuple( filter( in_output, self.full_output_structure ) )
            
    @property
    def output_shape(self):
        simdims =list(__sim_dims__.intersection( self.full_output_structure ))
        exp_dim_szs =[self.parameters[d].expanded for d in self.expanded_dims]
        return tuple( simdims + exp_dim_szs )
                