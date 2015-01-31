import numpy as np

def cs_remove_middle(alpha, threshold=10):
    """
    remove top and bottom `threshold` percentage of data in the cross section
    """
    lower_bounds = np.nanpercentile(alpha, threshold, axis=0)[np.newaxis,:]
    upper_bounds = np.nanpercentile(alpha, 100-threshold, axis=0)[np.newaxis,:]
    extremes = alpha.copy()
    extremes[np.logical_and(lower_bounds<extremes, 
                            extremes<upper_bounds)] = np.nan
    return extremes