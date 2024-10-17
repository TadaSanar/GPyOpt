# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from .base import AcquisitionBase
from ..util.general import get_quantiles

from ..util.general import samples_multidimensional_uniform #A

class AcquisitionEI_noisy(AcquisitionBase):
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

    def __init__(self, model, space, optimizer=None, cost_withGradients=None, jitter=0.01):
        self.optimizer = optimizer
        super(AcquisitionEI_noisy, self).__init__(model, space, optimizer, cost_withGradients=cost_withGradients)
        self.jitter = jitter

    @staticmethod
    def fromConfig(model, space, optimizer, cost_withGradients, config):
        return AcquisitionEI_noisy(model, space, optimizer, cost_withGradients, jitter=config['jitter'])

    def _compute_acq(self, x):
        """
        Computes the Expected Improvement per unit of cost
        """
        # Estimate the global minimum of the current model.
        bounds = self.space.get_bounds() #A
        points = create_ternary_grid(step = 0.005) #A
        fmin = self.model.predict(points)[0].min() #A
        
        m, s = self.model.predict(x, with_noise = False) #A Added with_noise = False
        #fmin = self.model.get_fmin() #A Removed this
        phi, Phi, u = get_quantiles(self.jitter, fmin, m, s)
        f_acqu = s * (u * Phi + phi)
        
        return f_acqu

    def _compute_acq_withGradients(self, x):
        """
        Computes the Expected Improvement and its derivative (has a very easy derivative!)
        """
        # Estimate the global minimum of the current model.
        bounds = self.space.get_bounds() #A
        points = create_ternary_grid(step = 0.01) #A
        fmin = self.model.predict(points)[0].min() #A
        
        #fmin = self.model.get_fmin() #A Removed this
        m, s, dmdx, dsdx = self.model.predict_withGradients(x, with_noise = False)  #A Added with_noise = False
        phi, Phi, u = get_quantiles(self.jitter, fmin, m, s)
        f_acqu = s * (u * Phi + phi)
        df_acqu = dsdx * phi - Phi * dmdx
        
        return f_acqu, df_acqu

def create_grid(dim = 3, domain_boundaries = [0.0,1.0], step = 0.005):
    """
    Generate a grid for the coordinates of datapoints that represent a 
    multidimensional cube of data with the desired spacing.

    Parameters
    ----------
    dim : int, optional
        Dimensionality of the datacube. The default is 3.
    domain_boundaries : [float], optional
        A list with two elements. The first element is the lower boundary of
        the domain (inclusive). The second element is the upper boundary of the
        domain (exclusive). The domain boundaries will be the same for each
        dimension. The default is [0.0,1.0].
    step : float, optional
        Step size for the points in the grid. The default is 0.005.

    Returns
    -------
    points : Numpy array
        Coordinates of the generated points. The shape of the array is
        (number of points, dim)

    """
    import numpy as np
    ### This grid is used for sampling+plotting the posterior mean and std_dv + acq function.
    a = np.arange(domain_boundaries[0], domain_boundaries[1], step)
    b = [a for i in range(dim)]
    
    grid_temp = np.meshgrid(*b, sparse=False)
    grid_temp_list = [grid_temp[i].ravel() for i in range(dim)]
    
    points = np.transpose(grid_temp_list)
    
    return points

def create_ternary_grid(step = 0.005):

    ### This grid is used for sampling+plotting the posterior mean and std_dv + acq function.
    points = create_grid(step = step)
    points = points[abs(points.sum(axis=1)-1) < (step - step/5)]
    
    return points