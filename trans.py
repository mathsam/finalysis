import numpy as np

def cs_remove_middle(alpha, threshold=10, padding=np.nan):
    """
    remove top and bottom `threshold` percentage of data in the cross section
    the removed data is filled with padding
    """
    lower_bounds = np.nanpercentile(alpha, threshold, axis=0)[np.newaxis,:]
    upper_bounds = np.nanpercentile(alpha, 100-threshold, axis=0)[np.newaxis,:]
    extremes = alpha.copy()
    extremes[np.logical_and(lower_bounds<extremes, 
                            extremes<upper_bounds)] = padding
    return extremes
    
def ts_delay(alpha, num_days):
    """
    delay alpha by `num_days`
    """
    delayed_signal = np.empty_like(alpha)
    if num_days >= 0:
        delayed_signal[:,num_days:]  = alpha[:,0:-num_days]
        delayed_signal[:,0:num_days] = np.nan
    else:
        delayed_signal[:,:num_days]  = alpha[:,-num_days:]
        delayed_signal[:,0:num_days] = np.nan
    return delayed_signal