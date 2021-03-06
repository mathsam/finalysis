import numpy as np
from scipy.stats import pearsonr
from pandas import Series, DataFrame
import pandas as pd

def xcorr(x, y, maxlags=20):
    """
    lagged correlation
    """
    if not (isinstance(x, np.ndarray) and 
            isinstance(y, np.ndarray)):
        raise TypeError('Inputs must be numpy.ndarray')
    
    if len(x)!=len(y):
        raise ValueError('Input variables of different lengths.')
            
    valid_index = np.logical_and(~np.isnan(x), ~np.isnan(y))
    valid_x = x[valid_index]
    valid_y = y[valid_index]

    if maxlags >= len(valid_x):
        raise ValueError('Max lag >= valid length of input signals')
        
    lags = np.arange(-maxlags, maxlags+1, 1)
    r    = np.empty_like(lags, dtype=float)
    pr   = np.empty_like(lags, dtype=float)
    for i, ilag in enumerate(lags):
        if ilag < 0:
            r[i], pr[i] = pearsonr(valid_x[:ilag],valid_y[-ilag:])
        elif ilag ==0:
            r[i], pr[i] = pearsonr(valid_x,valid_y)
        elif ilag>0:
            r[i], pr[i] = pearsonr(valid_x[ilag:],valid_y[:-ilag])
    return lags, r, pr
    
def xcorr2d(x2d, y2d, maxlags_i = 20, maxlags_j = 20):
    """
    correlation between x2d[i+ilag, j+jlag] and y2d[i, j]
    """
    lags = np.zeros((2*maxlags_i+1, 2*maxlags_j+1))
    r    = np.zeros((2*maxlags_i+1, 2*maxlags_j+1))
    pr   = np.zeros((2*maxlags_i+1, 2*maxlags_j+1))
    for i, ilag in enumerate(range(-maxlags_i, maxlags_i+1)):
        for j, jlag in enumerate(range(-maxlags_j, maxlags_j+1)):
            y2d_shifted = np.roll(y2d, ilag, 0)
            if ilag<0:
                y2d_shifted[ilag:,:] = np.nan
            elif ilag>0:
                y2d_shifted[:ilag,:] = np.nan
            y2d_shifted = np.roll(y2d_shifted, jlag, 1)
            if jlag<0:
                y2d_shifted[:,jlag:] = np.nan
            elif jlag>0:
                y2d_shifted[:,:jlag] = np.nan
            _, r[i,j], pr[i,j] = xcorr(x2d.flatten(), 
                                               y2d_shifted.flatten(), 0)
    lagsX, lagsY = np.meshgrid(np.arange(-maxlags_j, maxlags_j+1), 
                               np.arange(-maxlags_i, maxlags_i+1))
    return lagsX, lagsY, r, pr

def aggre_xcorr(x, y, maxlags=20, aggre_width=5):
    if not (isinstance(x, np.ndarray) and 
            isinstance(y, np.ndarray)):
        raise TypeError('Inputs must be numpy.ndarray')
    
    if len(x)!=len(y):
        raise ValueError('Input variables of different lengths.')
            
    valid_index = np.logical_and(~np.isnan(x), ~np.isnan(y))
    valid_x = x[valid_index]
    valid_y = y[valid_index]
    valid_x = smooth(valid_x, window_len=aggre_width, window='flat')
    valid_y = smooth(valid_y, window_len=aggre_width, window='flat')

    if maxlags*aggre_width >= len(valid_x):
        raise ValueError('maxlags*aggre_width >= valid length of input signals')

    lags = np.arange(-maxlags, maxlags+1, 1)
    r    = np.empty_like(lags, dtype=float)
    pr   = np.empty_like(lags, dtype=float)
    for i, ilag in enumerate(lags):
        if ilag < 0:
            r[i], pr[i] = pearsonr(valid_x[:ilag:aggre_width],valid_y[-ilag::aggre_width])
        elif ilag ==0:
            r[i], pr[i] = pearsonr(valid_x[::aggre_width],valid_y[::aggre_width])
        elif ilag>0:
            r[i], pr[i] = pearsonr(valid_x[ilag::aggre_width],valid_y[:-ilag:aggre_width])
    return lags, r, pr

def smooth(x, window_len=10, window='flat'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    import numpy as np    
    t = np.linspace(-2,2,0.1)
    x = np.sin(t)+np.random.randn(len(t))*0.1
    y = smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string   
    """

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"

    s=np.r_[2*x[0]-x[window_len:1:-1], x, 2*x[-1]-x[-1:-window_len:-1]]
    #print(len(s))
    
    if window == 'flat': #moving average
        w = np.ones(window_len,'d')
    else:
        w = getattr(np, window)(window_len)
    y = np.convolve(w/w.sum(), s, mode='same')
    return y[window_len-1:-window_len+1]
    
def quantile(time_series, numbins=10):
    """
    given a pandas Series `time_series`, returns a  Series with the same index
    but with the value corresponds to the quantile rank of the input Series
    """
    q, bins = pd.qcut(time_series, numbins, retbins=True)
    bins      = bins.astype(np.float64)
    bins[-1] += np.finfo(np.float64).eps
    quantile_array = np.full(time_series.shape, np.nan)
    binmid_array   = np.full(time_series.shape, np.nan)
    for i in range(0, len(bins)-1):
        ibin_index = (time_series.values>=bins[i]) & (time_series.values <bins[i+1])
        quantile_array[ibin_index] = i
        binmid_array[ibin_index]   = (bins[i]+bins[i+1])/2
    quantile_series = Series(data=quantile_array, index=time_series.index)
    binmid_series   = Series(data=binmid_array,   index=time_series.index)
    return quantile_series, binmid_series

def sort_by(x, y, numbins=15):
    x_y = DataFrame({'x': x, 'y': y})
    quantile_rank, binmid = quantile(x_y.x, numbins)
    x_y['quantile'] = quantile_rank
    x_y['bin']      = binmid
    x_y_bybin      = x_y.groupby('bin')
    y_bybin_mean = x_y_bybin.y.mean()
    return y_bybin_mean.index.values, y_bybin_mean.values
    
def time_to_lastshock(shock_signal):
    """
    given shock_signal consists of -1, 0, 1, turn it into the time to last shock
    signal. For example, for shock_signal [0, 0, 1, 0, 0, -1, 0, 0, 0], return 
    two arrays,
    time2pshock (time to positive shock): [nan, nan, 0, 1, 2, nan, nan, ...]
    time2nshock: [nan, nan, nan, nan, nan, 0, 1, 2, 3]
    """
    time2pshock = np.full(shock_signal.shape, np.nan)
    time2nshock = np.full(shock_signal.shape, np.nan)
    if shock_signal.ndim == 2:
        for k in range(0, shock_signal.shape[0]):
            p_days = np.nan
            n_days = np.nan
            for t in range(0, shock_signal.shape[1]):
                if shock_signal[k, t] == 1:
                    p_days = 0
                    n_days = np.nan
                elif shock_signal[k, t] == -1:
                    n_days = 0
                    p_days = np.nan
                else:
                    p_days += 1
                    n_days += 1
                if not np.isnan(p_days):
                    time2pshock[k, t] = p_days
                elif not np.isnan(n_days):
                    time2nshock[k, t] = n_days
        return time2pshock, time2nshock
        