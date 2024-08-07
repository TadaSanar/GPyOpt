# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from .base import AcquisitionBase
from ..util.general import get_quantiles

import GPy #A Added
import numpy as np

class AcquisitionEI_DF(AcquisitionBase):
    """
    Expected improvement acquisition function

    :param model: GPyOpt class of model
    :param space: GPyOpt class of domain
    :param optimizer: optimizer of the acquisition. Should be a GPyOpt optimizer
    :param cost_withGradients: function
    :param jitter: positive value to make the acquisition more explorative.

    .. Note:: allows to compute the Improvement per unit of cost

    """

    analytical_gradient_prediction = True

    def __init__(self, model, space, optimizer=None, cost_withGradients=None, jitter=0.01, ei_df_params=None, verbose = False):
        self.optimizer = optimizer
        super(AcquisitionEI_DF, self).__init__(model, space, optimizer, cost_withGradients=cost_withGradients)
        self.jitter = jitter
        self.verbose = verbose
        
        #A Added data fusion parameter handling
        
        if ei_df_params is None:
            
            # Default values.
            ei_df_params = {'p_beta': 0.025,
                         'p_midpoint': 0,
                         'df_model': None
                         }
        
        else:
            
            if 'df_model' in ei_df_params:
                
                # GPy GPRegression model.
                self.constraint_model = ei_df_params['df_model']
                
            else:
                
                raise Exception("Data fusion feature requires a dictionary of data fusion parameters with key 'df_model'. Provide 'None' or a GPy GPRegression model.")
        
        if 'p_beta' in ei_df_params:
            
            self.p_beta = ei_df_params['p_beta']
            
        else:
            
            # Default value.
            self.p_beta = 0.025

        if 'p_midpoint' in ei_df_params:
            
            self.p_midpoint = ei_df_params['p_midpoint']
            
        else:
            
            # Default value.
            self.p_beta = 0

        

    @staticmethod
    def fromConfig(model, space, optimizer, cost_withGradients, config):
        
        if 'ei_df_params' in config:
            
            ei_df_params = config['ei_df_params']
            
        else:
        
            ei_df_params = None
            
        if 'verbose' in config:
        
            verbose = config['verbose']
            
        return AcquisitionEI_DF(model, space, optimizer, cost_withGradients, jitter=config['jitter'], ei_df_params = ei_df_params)

    def _compute_acq(self, x):
        """
        Computes the Expected Improvement per unit of cost
        """
        m, s = self.model.predict(x)
        fmin = self.model.get_fmin()
        phi, Phi, u = get_quantiles(self.jitter, fmin, m, s)
        f_acqu = s * (u * Phi + phi)
        
        prob = calc_P(x, self.constraint_model, self.p_beta, self.p_midpoint) #A Added
        
        f_acqu = f_acqu * prob #A Added
        
        if self.verbose:
        
            message = 'Exploitation ' + str(s*u*Phi*prob) + ', exploration ' + str(s*phi*prob) # Added
            print(message)
        
        return f_acqu

    def _compute_acq_withGradients(self, x):
        """
        Computes the Expected Improvement and its derivative (has a very easy derivative!)
        """
        fmin = self.model.get_fmin()
        m, s, dmdx, dsdx = self.model.predict_withGradients(x)
        phi, Phi, u = get_quantiles(self.jitter, fmin, m, s)
        f_acqu = s * (u * Phi + phi)
        df_acqu = dsdx * phi - Phi * dmdx
        
        # A Added:
        if self.verbose and np.any(np.isnan(x)):
        
            message = 'x contains nan:\n ' + str(x)
            print(message)
        
        prob = calc_P(x, self.constraint_model, self.p_beta, self.p_midpoint) #A Added
        
        if self.verbose:
            
            message = 'Exploitation ' + str(s*u*Phi*prob) + ', exploration ' + str(s*phi*prob) # Added
            print(message)
            
            message = 'x='+str(x)+', acqu='+str(f_acqu)+', grad_acqu='+str(df_acqu) + ', P=' + str(prob)
            print(message)
        
        f_acqu = f_acqu * prob #A Added
        
        d_prob = calc_gradient_of_P(x, self.constraint_model, self.p_beta,
                                    self.p_midpoint)
        
        df_acqu = df_acqu * prob + f_acqu * d_prob
        
        if self.verbose:
            
            print('acqu_P='+str(f_acqu)+', grad_acqu_P='+str(df_acqu))
        
        return f_acqu, df_acqu

def calc_P(points, constraint_model, p_beta = 0.025, p_midpoint = 0):
    
    # GPy GPRegression model assumed.
    if constraint_model is not None:
    
        mean, _ = constraint_model.predict_noiseless(points)
        
        propability = inv_sigmoid(mean, p_midpoint, p_beta)
    
    else:
        
        # No data fusion data so no grounds for declaring any area less good.
        propability= np.ones(shape = (points.shape[0], 1))
        
    return propability

def inv_sigmoid(mean, p_midpoint, p_beta):
    
    # Inverted because the negative/lower values are assumed better than high
    # ones. This choice was made because the original application for data
    # fusion was DFT Gibbs free energies, where compositions with negative
    # energies are the ones that are stable.
    
    f = 1/(1+np.exp((mean-p_midpoint)/p_beta))
    
    return f
    
        
def calc_gradient_of_P(x, constraint_model, p_beta, p_midpoint):
    
    if constraint_model is None:
        
        g = np.zeros(x.shape)
        
    else:
        
        # Step size for numerical gradient.
        delta_x = constraint_model.kern.lengthscale/1000
        
        g = np.empty(x.shape)
        
        for i in range(x.shape[1]):
            
            x_l = x.copy()
            x_u = x.copy()
            
            x_l[:,i] = x_l[:,i] - delta_x/2
            x_u[:,i] = x_u[:,i] + delta_x/2
            
            p_l = calc_P(x_l, constraint_model, p_beta, p_midpoint)
            #p_c = calc_P(x, constraint_model, p_beta, p_midpoint)
            p_u = calc_P(x_u, constraint_model, p_beta, p_midpoint)
            
            g[:,i] =  np.ravel((p_u - p_l)/delta_x)
        
        return g
        
    
