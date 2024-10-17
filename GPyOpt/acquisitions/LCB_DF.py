# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from .base import AcquisitionBase
from ..util.general import get_quantiles

import GPy #A Added
import numpy as np

class AcquisitionLCB_DF(AcquisitionBase):
    """
    GP-Lower Confidence Bound acquisition function with constant exploration weight.
    See:
    
    Gaussian Process Optimization in the Bandit Setting: No Regret and Experimental Design
    Srinivas et al., Proc. International Conference on Machine Learning (ICML), 2010

    :param model: GPyOpt class of model
    :param space: GPyOpt class of domain
    :param optimizer: optimizer of the acquisition. Should be a GPyOpt optimizer
    :param cost_withGradients: function
    :param jitter: positive value to make the acquisition more explorative

    .. Note:: does not allow to be used with cost

    """

    analytical_gradient_prediction = True

    def __init__(self, model, space, optimizer=None, cost_withGradients=None, exploration_weight=None, ei_df_params=None, verbose = True):
        self.optimizer = optimizer
        super(AcquisitionLCB_DF, self).__init__(model, space, optimizer, cost_withGradients=None)
        if exploration_weight is None:
            raise Exception('\nEW is None.\n')
        self.exploration_weight = exploration_weight
        if cost_withGradients is not None:
            print('The set cost function is ignored! LCB acquisition does not make sense with cost.')  
            
        self.verbose = verbose
        
        #A Added data fusion parameter handling
        
        if ei_df_params is None:
            
            raise Exception("Data fusion feature requires a dictionary of data fusion parameters with key 'df_model'. Provide 'None' or a GPy GPRegression model.")
            
            ## Default values.
            #ei_df_params = {'p_beta': 0.025,
            #             'p_midpoint': 0,
            #             'df_model': None
            #             }
        
        else:
            
            if 'df_model' in ei_df_params:
                
                # GPy GPRegression model.
                self.constraint_model = ei_df_params['df_model']
                
            else:
                
                raise Exception("Data fusion feature requires a dictionary of data fusion parameters with key 'df_model'. Provide 'None' or a GPy GPRegression model.")
        
        if 'p_beta' in ei_df_params:
            
            self.p_beta = ei_df_params['p_beta']
            
        else:
            
            raise Exception("Data fusion params not forwarded.")
            
            # Default value.
            self.p_beta = 0.025

        if 'p_midpoint' in ei_df_params:
            
            self.p_midpoint = ei_df_params['p_midpoint']
            
        else:
            
            raise Exception("Data fusion params not forwarded.")
            
            # Default value.
            self.p_beta = 0

    def _compute_acq(self, x):
        """
        Computes the GP-Lower Confidence Bound 
        """
        
        m, s = self.model.predict(x)   
        f_acqu = -m + self.exploration_weight * s
        
        prob = calc_P(x, self.constraint_model, self.p_beta, self.p_midpoint) #A Added
        
        f_acqu = f_acqu * prob #A Added
        
        return f_acqu

    def _compute_acq_withGradients(self, x):
        """
        Computes the GP-Lower Confidence Bound and its derivative
        """
        m, s, dmdx, dsdx = self.model.predict_withGradients(x) 
        f_acqu = -m + self.exploration_weight * s       
        df_acqu = -dmdx + self.exploration_weight * dsdx
        
        prob = calc_P(x, self.constraint_model, self.p_beta, self.p_midpoint) #A Added
        
        f_acqu = f_acqu * prob #A Added
        
        d_prob = calc_gradient_of_P(x, self.constraint_model, self.p_beta,
                                    self.p_midpoint)
        
        df_acqu = df_acqu * prob + f_acqu * d_prob
        
        
        return f_acqu, df_acqu

def calc_P(points, constraint_model, p_beta = 0.025, p_midpoint = 0):
    
    # GPy GPRegression model assumed.
    if constraint_model is not None:
    
        mean, _ = constraint_model.predict_noiseless(points)
        
        propability = inv_sigmoid(mean, p_midpoint, p_beta)
    
    else:
        
        raise Exception("Data fusion feature requires a dictionary of data fusion parameters with key 'df_model'. Provide 'None' or a GPy GPRegression model.")
        
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