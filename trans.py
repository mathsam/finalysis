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
    
def ts_remove_middle(alpha, percent=10, lookback=10, padding=np.nan):
    """
    remove value if it is within [percent, 100-percent] percentile
    in previous `lookback` days containing today
    """
    percent = percent / 100.
    extremes    = np.empty_like(alpha)
    extremes[:] = np.nan
    total_time = alpha.shape[1]
    for t in range(lookback, total_time):
        #lower_bounds = np.nanpercentile(alpha[:,t-lookback:t], percent, axis=1)
        #upper_bounds = np.nanpercentile(alpha[:,t-lookback:t], 100-percent, axis=1)
        #ext_index = np.logical_or(alpha[:,t] < lower_bounds, 
        #                           alpha[:,t] > upper_bounds)
        sorted_alpha = np.sort(alpha[:,t-lookback:t+1], axis=1)
        ext_index = np.logical_or(
                      alpha[:,t] <= sorted_alpha[:, np.ceil(percent*lookback)-1],
                      alpha[:,t] >= sorted_alpha[:,-np.ceil(percent*lookback)])
        extremes[ext_index,t] = alpha[ext_index,t]
    return extremes
    
def ts_identify_shock(alpha, percent=10, lookback=10, padding=np.nan):
    """
    similar to ts_remove_middle, but replace the high and low anomaly by 1 and 
    -1
    """
    percent = percent / 100.
    extremes    = np.empty_like(alpha)
    extremes[:] = np.nan
    total_time = alpha.shape[1]
    for t in range(lookback, total_time):
        #lower_bounds = np.nanpercentile(alpha[:,t-lookback:t], percent, axis=1)
        #upper_bounds = np.nanpercentile(alpha[:,t-lookback:t], 100-percent, axis=1)
        #ext_index = np.logical_or(alpha[:,t] < lower_bounds, 
        #                           alpha[:,t] > upper_bounds)
        sorted_alpha = np.sort(alpha[:,t-lookback:t+1], axis=1)
        low_index = alpha[:,t] <= sorted_alpha[:, np.ceil(percent*lookback)-1]
        high_index= alpha[:,t] >= sorted_alpha[:,-np.ceil(percent*lookback)]
        extremes[low_index, t] = -1.
        extremes[high_index,t] = 1.
    return extremes

    
def ts_delay(alpha, num_days, padding=np.nan):
    """
    delay alpha by `num_days`
    """
    delayed_signal = np.empty_like(alpha)
    if num_days >= 0:
        delayed_signal[:,num_days:]  = alpha[:,0:-num_days]
        delayed_signal[:,0:num_days] = padding
    else:
        delayed_signal[:,:num_days] = alpha[:,-num_days:]
        delayed_signal[:,num_days:] = padding
    return delayed_signal