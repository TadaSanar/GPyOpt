# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from .base import AcquisitionBase
from ..util.general import get_quantiles

from EI_DF import calc_P, calc_gradient_of_P

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

    def __init__(self, model, space, optimizer=None, cost_withGradients=None, exploration_weight=2, ei_df_params=None, verbose = True):
        self.optimizer = optimizer
        super(AcquisitionLCB_DF, self).__init__(model, space, optimizer, cost_withGradients=None)
        self.exploration_weight = exploration_weight
        
        if cost_withGradients is not None:
            print('The set cost function is ignored! LCB acquisition does not make sense with cost.')  
            
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

